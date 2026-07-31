"""Pre-processing of ADI sequences: bad pixels, background, centring, selection.

This module is the entry point of the reduction chain: it turns a raw cube into
something :mod:`exoklip.klip` can work on. Nothing here is astrophysics --- it is
all instrumental hygiene, but every step of it directly limits the achievable
contrast:

* an uncorrected hot pixel behaves like a delta function and survives KLIP
  almost untouched (it is not correlated with the speckle field), so it shows up
  as a spurious point source;
* a residual background offset biases the annulus-wise mean subtraction of KLIP;
* a centring error of ``epsilon`` pixels rotates into an azimuthal smear of
  ``epsilon`` pixels after derotation and destroys the speckle correlation that
  KLIP relies on --- 0.1 px is the usual target;
* frames taken through clouds or during an AO loop opening only add noise.

Conventions
-----------
* Images are ``(y, x)``, cubes are ``(n_frames, y, x)``, ``float64`` internally.
* The frame centre is ``((ny - 1) / 2, (nx - 1) / 2)``
  (:func:`exoklip.core.frame_center`), for both parities.
* Angles are in degrees in every public API; position angles follow the package
  convention (0 = North = ``+y``, increasing towards East = ``-x``).
* Inputs are never modified: every function returns a new array.
* NaN means "missing data" and is preserved as such (see the notes of each
  function).

References
----------
Pueyo et al. 2015, ApJ 803, 31 -- Radon-transform centring of saturated /
coronagraphic stars.
Marois et al. 2006, ApJ 641, 556 -- ADI, azimuthal smearing budget.
Lafreniere et al. 2007, ApJ 660, 770 -- frame selection and the FWHM criterion.
Absil et al. 2013, A&A 559, L12 -- sub-pixel centring requirement for ADI.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage, optimize

from .core import dist_grid, frame_center
from .rotation import frame_shift

__all__ = [
    "bad_pixel_correction",
    "subtract_background",
    "find_star_center",
    "cube_recenter",
    "frame_selection",
    "temporal_binning",
]

logger = logging.getLogger(__name__)

#: Gaussian FWHM / sigma.
_FWHM_SIGMA: float = 2.0 * math.sqrt(2.0 * math.log(2.0))

#: 1 / Phi^-1(3/4): scales a median absolute deviation into a Gaussian sigma.
_MAD_TO_SIGMA: float = 1.4826

_CENTER_METHODS: tuple[str, ...] = ("gaussian", "radon", "symmetry")
_BACKGROUND_METHODS: tuple[str, ...] = ("median_annulus", "median", "mean", "plane")
_SELECTION_METRICS: tuple[str, ...] = ("corr", "flux", "fwhm")


# --------------------------------------------------------------------------- #
# private helpers -- validation
# --------------------------------------------------------------------------- #
def _as_frame(frame: ArrayLike, name: str = "frame") -> NDArray[np.float64]:
    """Validate a 2D input and return a private ``float64`` copy.

    Parameters
    ----------
    frame : array_like
        Candidate 2D image.
    name : str, optional
        Name used in the error message.

    Returns
    -------
    numpy.ndarray
        New ``(ny, nx)`` float64 array.

    Raises
    ------
    ValueError
        If ``frame`` is not 2D.
    """
    arr = np.array(frame, dtype=np.float64, copy=True)
    if arr.ndim != 2:
        raise ValueError(
            f"`{name}` must be a 2D image (y, x); got ndim={arr.ndim} with "
            f"shape {arr.shape}."
        )
    return arr


def _as_cube(cube: ArrayLike, name: str = "cube") -> NDArray[np.float64]:
    """Validate a 3D input and return a private ``float64`` copy.

    Parameters
    ----------
    cube : array_like
        Candidate 3D cube.
    name : str, optional
        Name used in the error message.

    Returns
    -------
    numpy.ndarray
        New ``(n_frames, ny, nx)`` float64 array.

    Raises
    ------
    ValueError
        If ``cube`` is not 3D.
    """
    arr = np.array(cube, dtype=np.float64, copy=True)
    if arr.ndim != 3:
        raise ValueError(
            f"`{name}` must be a 3D cube (n_frames, y, x); got ndim={arr.ndim} "
            f"with shape {arr.shape}."
        )
    return arr


def _as_cube_or_frame(
    data: ArrayLike, name: str = "cube"
) -> tuple[NDArray[np.float64], bool]:
    """Accept a 2D frame or a 3D cube, always return a 3D working copy.

    Parameters
    ----------
    data : array_like
        2D frame ``(y, x)`` or 3D cube ``(n_frames, y, x)``.
    name : str, optional
        Name used in the error message.

    Returns
    -------
    tuple
        ``(cube, was_2d)`` where ``cube`` is a new ``(n, ny, nx)`` float64 array
        and ``was_2d`` tells whether the caller passed a single frame (so that
        the public function can squeeze the output back).

    Raises
    ------
    ValueError
        If ``data`` is neither 2D nor 3D.
    """
    arr = np.array(data, dtype=np.float64, copy=True)
    if arr.ndim == 2:
        return arr[np.newaxis, ...], True
    if arr.ndim == 3:
        return arr, False
    raise ValueError(
        f"`{name}` must be 2D (y, x) or 3D (n_frames, y, x); got ndim={arr.ndim} "
        f"with shape {arr.shape}."
    )


def _check_positive(value: Any, name: str, strict: bool = True) -> float:
    """Validate a finite scalar, positive (``strict``) or non-negative.

    Parameters
    ----------
    value : object
        Candidate scalar.
    name : str
        Name used in the error message.
    strict : bool, optional
        If True require ``> 0``, else ``>= 0``.

    Returns
    -------
    float
        The value as a Python float.

    Raises
    ------
    ValueError
        If the value is not a finite scalar with the required sign.
    """
    try:
        val = float(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be a finite scalar; got {value!r}.") from exc
    if not math.isfinite(val) or (val <= 0 if strict else val < 0):
        raise ValueError(
            f"`{name}` must be a finite scalar "
            f"{'> 0' if strict else '>= 0'}; got {value!r}."
        )
    return val


def _check_int(value: Any, name: str, minimum: int) -> int:
    """Validate an integer argument with a lower bound.

    Parameters
    ----------
    value : object
        Candidate integer.
    name : str
        Name used in the error message.
    minimum : int
        Smallest accepted value.

    Returns
    -------
    int
        The validated integer.

    Raises
    ------
    ValueError
        If ``value`` is not an integer or is below ``minimum``.
    """
    try:
        ival = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"`{name}` must be an integer >= {minimum}; got {value!r}."
        ) from exc
    if ival != value or ival < minimum:
        raise ValueError(f"`{name}` must be an integer >= {minimum}; got {value!r}.")
    return ival


def _check_point(
    point: Sequence[float] | NDArray[np.float64] | None,
    name: str,
) -> tuple[float, float] | None:
    """Validate an optional ``(y, x)`` pixel position.

    Parameters
    ----------
    point : sequence of float or None
        Candidate ``(y, x)``.
    name : str
        Name used in the error message.

    Returns
    -------
    tuple of float or None
        ``(y, x)`` as floats, or None if ``point`` is None.

    Raises
    ------
    ValueError
        If ``point`` is not a finite 2-element sequence.
    """
    if point is None:
        return None
    arr = np.asarray(point, dtype=np.float64).ravel()
    if arr.size != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"`{name}` must be a finite 2-element sequence (y, x); got {point!r}."
        )
    return float(arr[0]), float(arr[1])


def _nan_reduce(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a ``np.nan*`` reduction without the "All-NaN slice" RuntimeWarning.

    Parameters
    ----------
    func : callable
        NaN-aware reduction (``np.nanmedian``, ``np.nanmean``, ...).
    *args, **kwargs
        Forwarded to ``func``.

    Returns
    -------
    Any
        Whatever ``func`` returns; all-NaN slices give NaN.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(*args, **kwargs)


def _robust_sigma(values: NDArray[np.float64]) -> float:
    """Median absolute deviation of the finite samples, scaled to a Gaussian sigma.

    Parameters
    ----------
    values : numpy.ndarray
        Samples, NaN allowed.

    Returns
    -------
    float
        ``1.4826 * MAD``, or 0.0 if fewer than two finite samples.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 0.0
    med = float(np.median(finite))
    return float(_MAD_TO_SIGMA * np.median(np.abs(finite - med)))


# --------------------------------------------------------------------------- #
# private helpers -- 2D Gaussian core fit
# --------------------------------------------------------------------------- #
def _gaussian_model(
    params: NDArray[np.float64],
    yy: NDArray[np.float64],
    xx: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Elliptical 2D Gaussian evaluated on scattered coordinates.

    Parameters
    ----------
    params : numpy.ndarray
        ``(amplitude, y0, x0, sigma_y, sigma_x, theta, offset)``; ``theta`` in
        radians, counter-clockwise, and rotates the ``sigma_x`` axis away from
        ``+x``.
    yy, xx : numpy.ndarray
        Pixel coordinates (same shape).

    Returns
    -------
    numpy.ndarray
        Model values, same shape as ``yy``.
    """
    amp, y0, x0, sig_y, sig_x, theta, offset = params
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dy = yy - y0
    dx = xx - x0
    u = cos_t * dx + sin_t * dy       # along the sigma_x axis
    v = -sin_t * dx + cos_t * dy      # along the sigma_y axis
    return offset + amp * np.exp(-0.5 * ((u / sig_x) ** 2 + (v / sig_y) ** 2))


def _fit_gaussian_core(
    frame: NDArray[np.float64],
    guess: tuple[float, float],
    fwhm: float,
    box: int | None = None,
    mask_radius: float | None = None,
) -> dict[str, Any]:
    """Fit an elliptical 2D Gaussian to the stellar core.

    Robust least squares (``scipy.optimize.least_squares``, ``loss='soft_l1'``)
    on a square box centred on ``guess``; NaN pixels and, if ``mask_radius`` is
    given, the pixels closer than ``mask_radius`` to ``guess`` (a saturated
    core) are excluded from the residual.

    Parameters
    ----------
    frame : numpy.ndarray
        2D image ``(y, x)``, float64.
    guess : tuple of float
        Initial ``(y, x)`` position of the core, in absolute pixel coordinates.
    fwhm : float
        Expected FWHM in pixels; sets the initial widths and the default box.
    box : int, optional
        Side of the fitting box in pixels. Default ``4 * fwhm`` rounded up to
        an odd value, at least 7.
    mask_radius : float, optional
        Pixels within this radius of ``guess`` are ignored (saturated core).

    Returns
    -------
    dict
        ``{'y', 'x', 'fwhm_y', 'fwhm_x', 'fwhm', 'theta', 'amplitude',
        'offset', 'success'}``; ``fwhm`` is the geometric mean of the two axes
        and ``theta`` is in degrees. On failure the position falls back to the
        flux-weighted centroid of the box and ``success`` is False.

    Notes
    -----
    A geometric-mean FWHM is used (same convention as
    :func:`exoklip.psf.fit_gaussian_psf`) because it is the width of the
    circular Gaussian with the same second-moment area.
    """
    ny, nx = frame.shape
    if box is None:
        box = int(round(4.0 * fwhm))
    box = max(7, int(box))
    if box % 2 == 0:
        box += 1
    half = box // 2

    gy, gx = guess
    y0i = int(round(min(max(gy, 0.0), ny - 1.0)))
    x0i = int(round(min(max(gx, 0.0), nx - 1.0)))
    ylo, yhi = max(0, y0i - half), min(ny, y0i + half + 1)
    xlo, xhi = max(0, x0i - half), min(nx, x0i + half + 1)

    sub = frame[ylo:yhi, xlo:xhi]
    yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
    yy = yy.astype(np.float64)
    xx = xx.astype(np.float64)

    valid = np.isfinite(sub)
    if mask_radius is not None and mask_radius > 0:
        valid &= np.hypot(yy - gy, xx - gx) >= float(mask_radius)

    failed = {
        "y": float(gy),
        "x": float(gx),
        "fwhm_y": float("nan"),
        "fwhm_x": float("nan"),
        "fwhm": float("nan"),
        "theta": float("nan"),
        "amplitude": float("nan"),
        "offset": float("nan"),
        "success": False,
    }
    if valid.sum() < 10:
        logger.warning(
            "Gaussian core fit: only %d usable pixel(s) in the %dx%d box around "
            "(%.2f, %.2f); returning the initial guess.",
            int(valid.sum()), box, box, gy, gx,
        )
        return failed

    data = sub[valid]
    ys = yy[valid]
    xs = xx[valid]

    offset0 = float(np.median(data))
    peak_idx = int(np.argmax(data))
    amp0 = max(float(data[peak_idx]) - offset0, 1e-12)
    y00 = float(ys[peak_idx]) if mask_radius is None else float(gy)
    x00 = float(xs[peak_idx]) if mask_radius is None else float(gx)
    sig0 = max(float(fwhm) / _FWHM_SIGMA, 0.5)

    p0 = np.array([amp0, y00, x00, sig0, sig0, 0.0, offset0], dtype=np.float64)
    lo = np.array(
        [0.0, ylo - 1.0, xlo - 1.0, 0.3, 0.3, -np.pi, -np.inf], dtype=np.float64
    )
    hi = np.array(
        [np.inf, yhi + 1.0, xhi + 1.0, 5.0 * box, 5.0 * box, np.pi, np.inf],
        dtype=np.float64,
    )
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

    scale = _robust_sigma(sub) or max(abs(amp0) * 1e-3, 1e-12)

    def _residual(p: NDArray[np.float64]) -> NDArray[np.float64]:
        return _gaussian_model(p, ys, xs) - data

    try:
        res = optimize.least_squares(
            _residual, p0, bounds=(lo, hi), loss="soft_l1", f_scale=scale,
            max_nfev=2000,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover
        logger.warning("Gaussian core fit failed (%s); using the centroid.", exc)
        res = None

    if res is None or not res.success:
        weights = np.clip(data - offset0, 0.0, None)
        if weights.sum() > 0:
            failed["y"] = float((weights * ys).sum() / weights.sum())
            failed["x"] = float((weights * xs).sum() / weights.sum())
        return failed

    amp, yf, xf, sig_y, sig_x, theta, offset = res.x
    fwhm_y = float(_FWHM_SIGMA * abs(sig_y))
    fwhm_x = float(_FWHM_SIGMA * abs(sig_x))
    return {
        "y": float(yf),
        "x": float(xf),
        "fwhm_y": fwhm_y,
        "fwhm_x": fwhm_x,
        "fwhm": float(math.sqrt(fwhm_y * fwhm_x)),
        "theta": float(np.degrees(theta)),
        "amplitude": float(amp),
        "offset": float(offset),
        "success": True,
    }


def _peak_guess(frame: NDArray[np.float64], fwhm: float) -> tuple[float, float]:
    """Position of the brightest structure, after smoothing at the PSF scale.

    Smoothing with a Gaussian of the PSF width keeps single hot pixels from
    winning the ``argmax``; run :func:`bad_pixel_correction` first for anything
    stronger than a few sigma.

    Parameters
    ----------
    frame : numpy.ndarray
        2D image, NaN allowed.
    fwhm : float
        PSF FWHM in pixels.

    Returns
    -------
    tuple of float
        ``(y, x)`` of the smoothed maximum.
    """
    filled = np.where(np.isfinite(frame), frame, 0.0)
    smoothed = ndimage.gaussian_filter(
        filled, sigma=max(float(fwhm) / _FWHM_SIGMA, 0.5), mode="nearest"
    )
    idx = int(np.argmax(smoothed))
    return float(idx // frame.shape[1]), float(idx % frame.shape[1])


# --------------------------------------------------------------------------- #
# private helpers -- centring metrics
# --------------------------------------------------------------------------- #
def _shift_budget(
    shape2d: tuple[int, int], d0: tuple[float, float], fwhm: float
) -> tuple[float, float]:
    """Search radius and usable field radius for the shift-based centring metrics.

    Parameters
    ----------
    shape2d : tuple of int
        ``(ny, nx)``.
    d0 : tuple of float
        Initial offset ``(dy, dx)`` of the guess with respect to the geometric
        centre, in pixels.
    fwhm : float
        PSF FWHM in pixels.

    Returns
    -------
    tuple of float
        ``(max_shift, r_field)``: the largest offset explored around ``d0`` and
        the radius of the region on which the metric is evaluated. ``r_field``
        is shrunk so that the compared / interpolated pixels always come from
        inside the detector, and floored at ``max(2 * fwhm, 4)``.
    """
    ny, nx = shape2d
    half = min(ny, nx) / 2.0
    max_shift = float(min(10.0, 0.25 * min(ny, nx)))
    budget = max_shift + float(np.hypot(*d0))
    r_field = half - 2.0 - budget
    floor = max(2.0 * float(fwhm), 4.0)
    if r_field < floor:
        r_field = min(max(floor, 2.0), max(half - 1.0, 1.0))
        logger.debug(
            "Centring: frame %dx%d is small, metric radius clamped to %.1f px.",
            ny, nx, r_field,
        )
    return max_shift, float(r_field)


def _symmetry_center(
    frame: NDArray[np.float64],
    fwhm: float,
    d0: tuple[float, float],
    mask_radius: float | None,
    order: int = 3,
) -> tuple[float, float]:
    """Centre by minimising the residual against the 180 deg rotated image.

    A rotation by 180 deg about ``c = c0 + d`` (``c0`` being the geometric
    centre ``(n - 1) / 2``) is *exactly* the array reversal
    ``frame[::-1, ::-1]`` followed by a translation of ``2 * d`` -- the rotation
    itself needs no interpolation. The minimised cost is

    .. math::
        C(d) = \\left\\langle \\left[ I(\\mathbf{r})
               - I_{180}(\\mathbf{r} - 2 d) \\right]^2 \\right\\rangle

    over a fixed central disc. It vanishes identically at the true centre of a
    centro-symmetric image (star + symmetric speckle halo) whatever the shape of
    the evaluation region, so the region choice cannot bias the result.
    Minimisation with Nelder-Mead (:func:`scipy.optimize.minimize`) after a 1 px
    coarse scan; for a peaked centro-symmetric source the cost is unimodal
    (for two Gaussians, ``C`` is proportional to ``1 - exp(-|d - d_true|^2 /
    sigma^2)``), so the scan only speeds convergence up.

    Parameters
    ----------
    frame : numpy.ndarray
        2D image, float64, NaN allowed.
    fwhm : float
        PSF FWHM in pixels.
    d0 : tuple of float
        Initial offset ``(dy, dx)`` from the geometric centre.
    mask_radius : float or None
        Pixels within this radius of the initial centre estimate are excluded
        (saturated core, coronagraphic mask).
    order : int, optional
        Spline order of the sub-pixel translation. Default 3.

    Returns
    -------
    tuple of float
        Absolute centre ``(cy, cx)`` in pixels.
    """
    ny, nx = frame.shape
    cy0, cx0 = frame_center(frame.shape)
    max_shift, r_field = _shift_budget((ny, nx), d0, fwhm)

    rad = dist_grid((ny, nx), center=(cy0 + d0[0], cx0 + d0[1]))
    region = rad <= r_field
    if mask_radius is not None and mask_radius > 0:
        region &= rad >= float(mask_radius)
    region &= np.isfinite(frame)
    n_min = max(20, int(0.02 * region.size))
    if int(region.sum()) < n_min:
        logger.warning(
            "Symmetry centring: only %d usable pixel(s) (need %d); returning "
            "the initial guess.", int(region.sum()), n_min,
        )
        return float(cy0 + d0[0]), float(cx0 + d0[1])

    rot180 = frame[::-1, ::-1]
    ref = frame[region]

    def _cost(d: NDArray[np.float64]) -> float:
        if np.hypot(d[0] - d0[0], d[1] - d0[1]) > max_shift:
            return 1e30
        shifted = frame_shift(
            rot180, 2.0 * d[0], 2.0 * d[1], order=order, mode="constant",
            cval=np.nan,
        )[region]
        diff = ref - shifted
        good = np.isfinite(diff)
        if int(good.sum()) < n_min:
            return 1e30
        return float(np.mean(diff[good] ** 2))

    steps = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    best_d = np.array(d0, dtype=np.float64)
    best_c = _cost(best_d)
    for s_y in steps:
        for s_x in steps:
            cand = np.array([d0[0] + s_y, d0[1] + s_x], dtype=np.float64)
            val = _cost(cand)
            if val < best_c:
                best_c, best_d = val, cand

    simplex = np.array(
        [best_d, best_d + [0.5, 0.0], best_d + [0.0, 0.5]], dtype=np.float64
    )
    res = optimize.minimize(
        _cost,
        best_d,
        method="Nelder-Mead",
        options={
            "xatol": 1e-3,
            "fatol": 1e-14,
            "maxiter": 600,
            "initial_simplex": simplex,
        },
    )
    d_opt = res.x if np.isfinite(res.fun) and res.fun < 1e29 else best_d
    logger.debug(
        "Symmetry centring: d=(%.4f, %.4f), cost=%.6g, nfev=%d.",
        d_opt[0], d_opt[1], float(res.fun), int(res.nfev),
    )
    return float(cy0 + d_opt[0]), float(cx0 + d_opt[1])


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def bad_pixel_correction(
    cube: ArrayLike,
    sigma: float = 5.0,
    size: int = 5,
    iterations: int = 2,
    protect_mask: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Replace outlying pixels by their local median.

    A pixel is an outlier when it departs from a median-filtered version of the
    frame by more than ``sigma`` robust standard deviations of that difference.
    Using a MAD-based sigma matters here: an ordinary standard deviation is
    inflated by the very outliers being looked for, so the threshold drifts
    upward and the brightest defects survive.

    Parameters
    ----------
    cube : array_like
        ``(n_frames, y, x)`` cube, or a single ``(y, x)`` frame.
    sigma : float, default 5.0
        Rejection threshold in robust standard deviations.
    size : int, default 5
        Side of the median-filter footprint in pixels. Keep it larger than the
        defects and smaller than the PSF structure you want to preserve.
    iterations : int, default 2
        Passes. A second pass catches the neighbours of a strong defect, whose
        local median was itself contaminated during the first pass.
    protect_mask : array_like, optional
        Boolean ``(y, x)`` mask of pixels that are never corrected. **Use it for
        the stellar core**, which is a legitimate several-hundred-sigma outlier
        and would otherwise be flattened.

    Returns
    -------
    ndarray
        Corrected copy, same shape as the input.
    """
    work, was_frame = _as_cube_or_frame(cube)
    sigma_v = _check_positive(sigma, "sigma")
    size_i = _check_int(size, "size", 3)
    n_iter = _check_int(iterations, "iterations", 1)

    protect = None
    if protect_mask is not None:
        protect = np.asarray(protect_mask, dtype=bool)
        if protect.shape != work.shape[1:]:
            raise ValueError(
                f"protect_mask has shape {protect.shape} but the frames are "
                f"{work.shape[1:]}."
            )

    total = 0
    for index in range(work.shape[0]):
        frame = work[index]
        for _ in range(n_iter):
            filled = np.where(np.isfinite(frame), frame, 0.0)
            smooth = ndimage.median_filter(filled, size=size_i, mode="nearest")
            difference = frame - smooth
            scatter = _robust_sigma(difference)
            if not np.isfinite(scatter) or scatter <= 0:
                break
            bad = np.abs(difference) > sigma_v * scatter
            bad |= ~np.isfinite(frame)
            if protect is not None:
                bad &= ~protect
            if not np.any(bad):
                break
            # A star's core is a legitimate several-hundred-sigma outlier and
            # will be flattened like any defect. Silently destroying the
            # photometric reference is worse than a noisy frame, so say so.
            if protect is None and np.isfinite(frame).any():
                brightest = np.unravel_index(np.nanargmax(frame), frame.shape)
                if bad[brightest]:
                    warnings.warn(
                        "bad_pixel_correction is replacing the brightest pixel of "
                        f"frame {index}. If that is the stellar core, pass "
                        "protect_mask to exclude it, otherwise the photometric "
                        "reference of the whole reduction is destroyed.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            frame = np.where(bad, smooth, frame)
            total += int(bad.sum())
        work[index] = frame

    logger.info("Bad-pixel correction replaced %d pixel values.", total)
    return work[0] if was_frame else work


def subtract_background(
    cube: ArrayLike,
    method: str = "median_annulus",
    r_in: float | None = None,
    center: Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """Remove an additive background pedestal, frame by frame.

    Parameters
    ----------
    cube : array_like
        Cube or single frame.
    method : {'median_annulus', 'median', 'mean', 'plane'}
        ``'median_annulus'`` takes the median outside ``r_in``, keeping the
        stellar halo out of the estimate; it is the safe default. ``'plane'``
        fits and removes a tilted plane, for a detector with a gradient.
    r_in : float, optional
        Inner radius for ``'median_annulus'``. Defaults to 40 % of the largest
        inscribed radius.
    center : sequence of float, optional
        Stellar centre.

    Returns
    -------
    ndarray
        Background-subtracted copy.
    """
    work, was_frame = _as_cube_or_frame(cube)
    if method not in _BACKGROUND_METHODS:
        raise ValueError(
            f"unknown method {method!r}; expected one of {list(_BACKGROUND_METHODS)}."
        )

    shape2d = work.shape[1:]
    ctr = frame_center(shape2d) if center is None else _check_point(center, "center")

    if method == "median_annulus":
        radial = dist_grid(shape2d, center=ctr)
        limit = float(
            min(ctr[0], ctr[1], shape2d[0] - 1 - ctr[0], shape2d[1] - 1 - ctr[1])
        )
        inner = 0.4 * limit if r_in is None else _check_positive(r_in, "r_in")
        region = radial >= inner
        if not np.any(region):
            raise ValueError(
                f"r_in={inner:.1f} px leaves no pixel in the background annulus "
                f"of a {shape2d[0]}x{shape2d[1]} frame."
            )
        for i in range(work.shape[0]):
            work[i] -= _nan_reduce(np.nanmedian, work[i][region])
    elif method in ("median", "mean"):
        reducer = np.nanmedian if method == "median" else np.nanmean
        for i in range(work.shape[0]):
            work[i] -= _nan_reduce(reducer, work[i])
    else:  # plane
        yy, xx = np.indices(shape2d, dtype=np.float64)
        for i in range(work.shape[0]):
            good = np.isfinite(work[i])
            if int(good.sum()) < 3:
                continue
            design = np.column_stack(
                [yy[good], xx[good], np.ones(int(good.sum()), dtype=np.float64)]
            )
            coef, *_ = np.linalg.lstsq(design, work[i][good], rcond=None)
            work[i] -= coef[0] * yy + coef[1] * xx + coef[2]

    return work[0] if was_frame else work


def _radon_center(
    frame: NDArray[np.float64],
    fwhm: float,
    d0: tuple[float, float],
    mask_radius: float | None,
) -> tuple[float, float]:
    """Centre by maximising the Radon cost of Pueyo et al. (2015).

    Sums the image along lines passing through a candidate centre, over a range
    of angles. The true centre maximises that sum, because the speckle halo and
    the diffraction pattern are point-symmetric about the star. Unlike a
    Gaussian fit this never looks at the core, so a saturated or occulted star
    is no obstacle.

    Parameters
    ----------
    frame : numpy.ndarray
        2D image.
    fwhm : float
        PSF FWHM in pixels.
    d0 : tuple of float
        Initial offset ``(dy, dx)`` from the geometric centre, matching the
        convention of :func:`_symmetry_center` and :func:`_shift_budget`.
    mask_radius : float or None
        Pixels within this radius of the initial estimate are excluded.

    Returns
    -------
    tuple of float
        Absolute centre ``(cy, cx)`` in pixels.
    """
    origin = frame_center(frame.shape)
    guess = (origin[0] + d0[0], origin[1] + d0[1])
    radius, field = _shift_budget(frame.shape, d0, fwhm)
    filled = np.where(np.isfinite(frame), frame, 0.0)
    if mask_radius is not None and mask_radius > 0:
        radial = dist_grid(frame.shape, center=guess)
        filled = np.where(radial <= mask_radius, 0.0, filled)

    angles = np.deg2rad(np.arange(0.0, 180.0, 15.0))
    samples = np.linspace(-field, field, max(int(2 * field), 32))
    offsets = np.stack(
        [np.outer(np.cos(angles), samples), np.outer(np.sin(angles), samples)]
    )  # (2, n_angles, n_samples)

    def cost(point: NDArray[np.float64]) -> float:
        ys = point[0] + offsets[0]
        xs = point[1] + offsets[1]
        line = ndimage.map_coordinates(
            filled, [ys.ravel(), xs.ravel()], order=1, mode="constant"
        )
        return -float(np.sum(line**2))

    start = np.array(guess, dtype=np.float64)
    best_point, best_value = start, cost(start)
    step = max(radius / 4.0, 0.5)
    for dy in (-step, 0.0, step):
        for dx in (-step, 0.0, step):
            trial = start + np.array([dy, dx])
            value = cost(trial)
            if value < best_value:
                best_value, best_point = value, trial

    result = optimize.minimize(
        cost,
        best_point,
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-9, "maxiter": 400},
    )
    point = result.x if np.isfinite(result.fun) else best_point
    return float(point[0]), float(point[1])


def find_star_center(
    frame: ArrayLike,
    fwhm: float,
    method: str = "gaussian",
    mask_radius: float | None = None,
    guess: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Locate the star to sub-pixel accuracy.

    Centring accuracy propagates into every later step: ADI derotation happens
    about the assumed centre, so an error there smears the entire field along an
    arc. Absil et al. (2013) put the requirement below roughly 0.1 pixel.

    Parameters
    ----------
    frame : array_like
        Single 2D frame.
    fwhm : float
        Instrumental FWHM in pixels.
    method : {'gaussian', 'radon', 'symmetry'}
        ``'gaussian'`` fits the core: accurate, but only valid when the core is
        unsaturated and unocculted. ``'radon'`` (Pueyo et al. 2015) and
        ``'symmetry'`` both work on a saturated or coronagraphic star, because
        they exploit the point symmetry of the halo rather than the core.
    mask_radius : float, optional
        Radius of a central region excluded from the metric. Set it beyond the
        saturated area when using the halo-based methods.
    guess : sequence of float, optional
        Starting ``(y, x)``. Defaults to the smoothed brightest structure.

    Returns
    -------
    tuple of float
        ``(cy, cx)``.
    """
    arr = _as_frame(frame)
    fwhm_v = _check_positive(fwhm, "fwhm")
    if method not in _CENTER_METHODS:
        raise ValueError(
            f"unknown method {method!r}; expected one of {list(_CENTER_METHODS)}."
        )
    point = _check_point(guess, "guess") or _peak_guess(arr, fwhm_v)

    if method == "gaussian":
        fit = _fit_gaussian_core(arr, point, fwhm_v, mask_radius=mask_radius)
        return float(fit["y"]), float(fit["x"])

    # The shift-based metrics parametrise the centre as an OFFSET from the
    # geometric centre, not as an absolute position. Passing an absolute
    # position here silently eats the whole field budget in _shift_budget and
    # the metric degenerates to "return the initial guess".
    origin = frame_center(arr.shape)
    offset = (point[0] - origin[0], point[1] - origin[1])

    if method == "symmetry":
        return _symmetry_center(arr, fwhm_v, offset, mask_radius)
    return _radon_center(arr, fwhm_v, offset, mask_radius)


def cube_recenter(
    cube: ArrayLike,
    fwhm: float,
    method: str = "symmetry",
    mask_radius: float | None = None,
    center: Sequence[float] | None = None,
    order: int = 3,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shift every frame so that the star lands on a common centre.

    Parameters
    ----------
    cube : array_like
        ``(n_frames, y, x)``.
    fwhm : float
        Instrumental FWHM in pixels.
    method : {'symmetry', 'gaussian', 'radon'}, default 'symmetry'
        Centring metric, see :func:`find_star_center`.
    mask_radius : float, optional
        Central exclusion radius for the metric.
    center : sequence of float, optional
        Target position. Defaults to the geometric frame centre.
    order : int, default 3
        Spline order used for the sub-pixel shift.
    **kwargs
        Forwarded to :func:`find_star_center`.

    Returns
    -------
    tuple
        ``(recentred_cube, shifts)``, where ``shifts`` is ``(n_frames, 2)`` in
        ``(dy, dx)``: the correction applied to each frame.
    """
    work = _as_cube(cube)
    fwhm_v = _check_positive(fwhm, "fwhm")
    target = (
        frame_center(work.shape[1:]) if center is None else _check_point(center, "center")
    )

    shifts = np.zeros((work.shape[0], 2), dtype=np.float64)
    for i in range(work.shape[0]):
        cy, cx = find_star_center(
            work[i], fwhm_v, method=method, mask_radius=mask_radius, **kwargs
        )
        dy, dx = target[0] - cy, target[1] - cx
        shifts[i] = (dy, dx)
        if abs(dy) > 1e-3 or abs(dx) > 1e-3:
            work[i] = frame_shift(work[i], dy, dx, order=order, cval=np.nan)

    magnitude = np.hypot(shifts[:, 0], shifts[:, 1])
    logger.info(
        "Recentred %d frames: median shift %.3f px, maximum %.3f px.",
        work.shape[0],
        float(np.median(magnitude)),
        float(magnitude.max()),
    )
    if magnitude.max() > 0.25 * min(work.shape[1:]):
        warnings.warn(
            f"a recentring shift of {magnitude.max():.1f} px exceeds a quarter of "
            "the frame; the centring metric has probably locked onto the wrong "
            "structure.",
            RuntimeWarning,
            stacklevel=2,
        )
    return work, shifts


def frame_selection(
    cube: ArrayLike,
    fwhm: float,
    metric: str = "corr",
    percentile: float = 90.0,
    center: Sequence[float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Keep the best frames of a sequence.

    Observing conditions vary through a sequence, and a handful of frames taken
    through poor correction can dominate the residual noise. Discarding them
    usually improves the final contrast even though it reduces the total
    integration time (Lafreniere et al. 2007).

    Parameters
    ----------
    cube : array_like
        ``(n_frames, y, x)``.
    fwhm : float
        Instrumental FWHM in pixels.
    metric : {'corr', 'flux', 'fwhm'}
        ``'corr'``: Pearson correlation with the median frame over the central
        region, the general-purpose choice. ``'flux'``: core aperture flux.
        ``'fwhm'``: fitted core width, where *smaller is better*.
    percentile : float, default 90.0
        Percentage of frames to keep, in ``(0, 100]``.
    center : sequence of float, optional
        Stellar centre.

    Returns
    -------
    tuple
        ``(selected_cube, indices)``. The indices are in ascending order, so the
        temporal ordering — and therefore the correspondence with the
        parallactic angles — is preserved. Slice your angle array with them.
    """
    work = _as_cube(cube)
    fwhm_v = _check_positive(fwhm, "fwhm")
    if metric not in _SELECTION_METRICS:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {list(_SELECTION_METRICS)}."
        )
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"percentile must be in (0, 100]; got {percentile!r}.")

    shape2d = work.shape[1:]
    ctr = frame_center(shape2d) if center is None else _check_point(center, "center")
    radial = dist_grid(shape2d, center=ctr)
    region = radial <= max(8.0 * fwhm_v, 3.0)

    if metric == "corr":
        reference = _nan_reduce(np.nanmedian, work, axis=0)[region]
        reference = reference - np.nanmean(reference)
        norm = float(np.sqrt(np.nansum(reference**2)))
        scores = np.empty(work.shape[0], dtype=np.float64)
        for i, frame in enumerate(work):
            values = frame[region]
            values = values - np.nanmean(values)
            denominator = float(np.sqrt(np.nansum(values**2))) * norm
            scores[i] = float(np.nansum(values * reference) / denominator) if denominator > 0 else 0.0
    elif metric == "flux":
        core = radial <= fwhm_v / 2.0
        scores = np.array([_nan_reduce(np.nansum, f[core]) for f in work], dtype=np.float64)
    else:  # 'fwhm': smaller is better, hence the sign flip
        scores = np.array(
            [
                -float(_fit_gaussian_core(f, _peak_guess(f, fwhm_v), fwhm_v)["fwhm"])
                for f in work
            ],
            dtype=np.float64,
        )

    scores = np.where(np.isfinite(scores), scores, -np.inf)
    n_keep = max(1, int(round(work.shape[0] * float(percentile) / 100.0)))
    keep = np.sort(np.argsort(scores)[::-1][:n_keep]).astype(np.intp)
    logger.info(
        "Frame selection (%s) kept %d of %d frames.", metric, keep.size, work.shape[0]
    )
    return work[keep], keep


def temporal_binning(
    cube: ArrayLike,
    angles: ArrayLike,
    n_bin: int,
    fwhm: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average consecutive frames to cut the cost of a reduction.

    Binning trades azimuthal resolution for speed: within a bin the field keeps
    rotating, so a companion is smeared along an arc. The warning below fires
    once that arc exceeds one FWHM at the edge of the field, which is where the
    smearing starts costing real sensitivity.

    Parameters
    ----------
    cube : array_like
        ``(n_frames, y, x)``.
    angles : array_like
        ``(n_frames,)`` parallactic angles in degrees.
    n_bin : int
        Frames per bin. A trailing partial bin is kept, averaged over the frames
        it actually contains.
    fwhm : float, optional
        Instrumental FWHM, which enables the smearing check.

    Returns
    -------
    tuple
        ``(binned_cube, binned_angles)``.
    """
    work = _as_cube(cube)
    ang = np.asarray(angles, dtype=np.float64).ravel()
    if ang.size != work.shape[0]:
        raise ValueError(
            f"angles has {ang.size} entries but the cube has {work.shape[0]} "
            "frames; there must be exactly one parallactic angle per frame."
        )
    n = _check_int(n_bin, "n_bin", 1)
    if n == 1:
        return work, ang

    n_out = int(math.ceil(work.shape[0] / n))
    binned = np.empty((n_out, *work.shape[1:]), dtype=np.float64)
    binned_angles = np.empty(n_out, dtype=np.float64)

    worst = 0.0
    for i in range(n_out):
        chunk = slice(i * n, min((i + 1) * n, work.shape[0]))
        binned[i] = _nan_reduce(np.nanmean, work[chunk], axis=0)
        piece = ang[chunk]
        binned_angles[i] = float(np.mean(piece))
        worst = max(worst, float(piece.max() - piece.min()))

    if fwhm is not None and worst > 0:
        radius = float(min(work.shape[1:])) / 2.0
        arc = float(np.deg2rad(worst) * radius)
        if arc > fwhm:
            warnings.warn(
                f"binning {n} frames spans up to {worst:.1f} deg, smearing a "
                f"companion at the field edge over {arc:.1f} px, more than the "
                f"{fwhm:.1f} px FWHM. Reduce n_bin to keep the outer field sharp.",
                RuntimeWarning,
                stacklevel=2,
            )

    logger.info("Binned %d frames into %d.", work.shape[0], n_out)
    return binned, binned_angles
