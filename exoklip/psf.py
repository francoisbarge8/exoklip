"""Point-spread function models, FWHM measurement and photometric normalisation.

This module owns the **photometric reference** of the whole package. Every
contrast in ``exoklip`` is expressed as a ratio of aperture fluxes, and the
aperture in question is defined here: a circular aperture of *diameter* equal to
the FWHM, centred on the source. :func:`normalize_psf` rescales an off-axis PSF
so that the flux inside that aperture is exactly 1, which is what makes
``flux`` arguments elsewhere (``inject.inject_companion``, ``simulate``,
``metrics.contrast_curve``) directly interpretable as contrasts once multiplied
by the stellar aperture flux.

Conventions follow :mod:`exoklip.core`: images are ``(y, x)``, the centre is
``(n - 1) / 2`` on each axis, and angles are in degrees.

References
----------
Airy, G. B. 1835, Transactions of the Cambridge Philosophical Society, 5, 283
Moffat, A. F. J. 1969, A&A, 3, 455
Born, M. & Wolf, E. 1999, Principles of Optics, 7th ed., section 8.5.2
    (annular aperture diffraction pattern)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.special import j1

from .core import frame_center, pad_or_crop
from .rotation import frame_shift

__all__ = [
    "gaussian_2d",
    "moffat_2d",
    "airy_2d",
    "fwhm_lambda_over_d",
    "fit_gaussian_psf",
    "normalize_psf",
    "create_synthetic_psf",
]

logger = logging.getLogger(__name__)

#: FWHM of a Gaussian in units of its standard deviation.
_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

#: First zero of ``2 J1(v) / v``, i.e. the edge of the Airy disc.
_AIRY_FIRST_ZERO = 3.8317059702075125


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _as_frame(frame: ArrayLike, name: str = "frame") -> NDArray[np.float64]:
    """Validate and convert a 2D image to a fresh float64 array."""
    arr = np.array(frame, dtype=np.float64, copy=True)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D image (y, x); got shape {arr.shape} "
            f"with ndim={arr.ndim}."
        )
    if arr.size == 0:
        raise ValueError(f"{name} is empty (shape {arr.shape}).")
    return arr


def _as_shape(shape: Any) -> tuple[int, int]:
    """Coerce ``shape`` to a 2D ``(ny, nx)`` tuple of positive ints."""
    if isinstance(shape, (int, np.integer)):
        shape = (int(shape), int(shape))
    elif isinstance(shape, np.ndarray):
        shape = shape.shape
    if not isinstance(shape, Sequence) or len(shape) < 2:
        raise ValueError(
            f"shape must be an int or a sequence of at least 2 ints; got {shape!r}."
        )
    ny, nx = int(shape[-2]), int(shape[-1])
    if ny <= 0 or nx <= 0:
        raise ValueError(f"shape must be strictly positive; got ({ny}, {nx}).")
    return ny, nx


def _check_center(
    center: tuple[float, float] | None, shape: tuple[int, int]
) -> tuple[float, float]:
    if center is None:
        return frame_center(shape)
    if len(center) != 2:
        raise ValueError(f"center must be a (cy, cx) pair; got {center!r}.")
    return float(center[0]), float(center[1])


def _check_positive(value: float, name: str) -> float:
    val = float(value)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive; got {value!r}.")
    return val


def _radial_grid(
    shape: tuple[int, int], center: tuple[float, float]
) -> NDArray[np.float64]:
    cy, cx = center
    yy = np.arange(shape[0], dtype=np.float64) - cy
    xx = np.arange(shape[1], dtype=np.float64) - cx
    return np.hypot(yy[:, None], xx[None, :])


def _aperture_weights(
    shape: tuple[int, int],
    cy: float,
    cx: float,
    radius: float,
    subsample: int = 16,
) -> NDArray[np.float64]:
    """Per-pixel overlap fraction with a circular aperture.

    Pixels entirely inside the circle get a weight of exactly 1, pixels entirely
    outside get 0, and boundary pixels are resolved by ``subsample x subsample``
    sub-sampling (accurate to better than ~0.1 % of the aperture area).

    This is a private duplicate of :func:`exoklip.metrics.aperture_flux`'s
    weighting, kept here so that :mod:`exoklip.psf` stays free of a circular
    import (``metrics`` depends on ``adi``, which depends on ``psf``).
    """
    ny, nx = shape
    yy = np.arange(ny, dtype=np.float64) - cy
    xx = np.arange(nx, dtype=np.float64) - cx
    dist = np.hypot(yy[:, None], xx[None, :])

    half_diag = np.sqrt(2.0) / 2.0
    weights = np.zeros(shape, dtype=np.float64)
    weights[dist <= radius - half_diag] = 1.0

    boundary = (dist > radius - half_diag) & (dist < radius + half_diag)
    if not np.any(boundary):
        return weights

    # Sub-pixel offsets at the centres of a subsample x subsample grid.
    step = 1.0 / subsample
    offs = (np.arange(subsample, dtype=np.float64) + 0.5) * step - 0.5
    dy_sub, dx_sub = np.meshgrid(offs, offs, indexing="ij")
    dy_sub = dy_sub.ravel()
    dx_sub = dx_sub.ravel()

    by, bx = np.nonzero(boundary)
    py = by.astype(np.float64) - cy
    px = bx.astype(np.float64) - cx
    # (n_boundary, subsample**2)
    sub_dist = np.hypot(py[:, None] + dy_sub[None, :], px[:, None] + dx_sub[None, :])
    weights[by, bx] = np.mean(sub_dist <= radius, axis=1)
    return weights


def _aperture_sum(
    frame: NDArray[np.float64], cy: float, cx: float, radius: float
) -> float:
    """Sum of ``frame`` inside a circular aperture, NaN treated as zero."""
    weights = _aperture_weights(frame.shape, cy, cx, radius)
    finite = np.isfinite(frame)
    return float(np.sum(np.where(finite, frame, 0.0) * weights))


# --------------------------------------------------------------------------- #
# Analytic PSF models
# --------------------------------------------------------------------------- #
def gaussian_2d(
    shape: Any,
    fwhm: float,
    center: tuple[float, float] | None = None,
    amplitude: float = 1.0,
    theta: float = 0.0,
    fwhm_y: float | None = None,
) -> NDArray[np.float64]:
    """Elliptical 2D Gaussian.

    Parameters
    ----------
    shape : int or tuple of int
        Output shape ``(ny, nx)``. A scalar means a square frame.
    fwhm : float
        Full width at half maximum along the major axis, in pixels. When
        ``fwhm_y`` is ``None`` the profile is circular and this is *the* FWHM.
    center : tuple of float, optional
        ``(cy, cx)``. Defaults to :func:`exoklip.core.frame_center`.
    amplitude : float, default 1.0
        Peak value (the model is *not* flux-normalised; use
        :func:`normalize_psf` for that).
    theta : float, default 0.0
        Position angle of the ``fwhm`` axis, in degrees, counter-clockwise from
        the ``+x`` axis.
    fwhm_y : float, optional
        FWHM along the orthogonal axis. Defaults to ``fwhm`` (circular).

    Returns
    -------
    ndarray
        ``(ny, nx)`` float64 image.
    """
    ny, nx = _as_shape(shape)
    cy, cx = _check_center(center, (ny, nx))
    fwhm_x = _check_positive(fwhm, "fwhm")
    fwhm_min = fwhm_x if fwhm_y is None else _check_positive(fwhm_y, "fwhm_y")

    sigma_x = fwhm_x / _SIGMA_TO_FWHM
    sigma_y = fwhm_min / _SIGMA_TO_FWHM

    yy = np.arange(ny, dtype=np.float64)[:, None] - cy
    xx = np.arange(nx, dtype=np.float64)[None, :] - cx

    th = np.deg2rad(float(theta))
    cos_t, sin_t = np.cos(th), np.sin(th)
    x_rot = xx * cos_t + yy * sin_t
    y_rot = -xx * sin_t + yy * cos_t

    exponent = -0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)
    return float(amplitude) * np.exp(exponent)


def moffat_2d(
    shape: Any,
    fwhm: float,
    beta: float = 2.5,
    center: tuple[float, float] | None = None,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Circular Moffat profile (Moffat 1969).

    ``I(r) = amplitude * (1 + (r / alpha)^2) ** (-beta)`` with
    ``alpha = fwhm / (2 * sqrt(2 ** (1 / beta) - 1))``.

    The Moffat profile has heavier wings than a Gaussian and is the usual
    empirical description of a seeing-limited or partially-corrected PSF.

    Parameters
    ----------
    shape : int or tuple of int
        Output shape.
    fwhm : float
        Full width at half maximum in pixels.
    beta : float, default 2.5
        Wing index. ``beta -> inf`` recovers a Gaussian; typical AO-corrected
        values are 2-4.
    center : tuple of float, optional
        ``(cy, cx)``.
    amplitude : float, default 1.0
        Peak value.
    """
    ny, nx = _as_shape(shape)
    cy, cx = _check_center(center, (ny, nx))
    fwhm_v = _check_positive(fwhm, "fwhm")
    beta_v = _check_positive(beta, "beta")

    alpha = fwhm_v / (2.0 * np.sqrt(2.0 ** (1.0 / beta_v) - 1.0))
    r = _radial_grid((ny, nx), (cy, cx))
    return float(amplitude) * (1.0 + (r / alpha) ** 2) ** (-beta_v)


def _airy_amplitude(v: NDArray[np.float64], obscuration: float) -> NDArray[np.float64]:
    """Diffraction amplitude of an annular pupil, normalised to 1 at ``v = 0``.

    ``A(v) = [2 J1(v) / v - eps^2 * 2 J1(eps v) / (eps v)] / (1 - eps^2)``

    Note the ``eps^2`` factor: it is what makes ``A(0) = 1``, since both
    ``2 J1(u) / u`` terms tend to 1 and ``(1 - eps^2) / (1 - eps^2) = 1``.
    (Born & Wolf 1999, section 8.5.2.)
    """
    eps = float(obscuration)

    def _jinc(u: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.ones_like(u)
        nz = u != 0.0
        out[nz] = 2.0 * j1(u[nz]) / u[nz]
        return out

    if eps == 0.0:
        return _jinc(v)
    return (_jinc(v) - eps**2 * _jinc(eps * v)) / (1.0 - eps**2)


@lru_cache(maxsize=32)
def _airy_fwhm_factor(obscuration: float) -> float:
    """FWHM of an Airy pattern in units of ``lambda / D``.

    Computed numerically rather than hard-coded, because a central obscuration
    narrows the core: the familiar 1.028 holds only for an unobscured pupil.
    """

    def half_max(v: float) -> float:
        amp = _airy_amplitude(np.array([v], dtype=np.float64), obscuration)[0]
        return amp**2 - 0.5

    # The half-maximum radius always lies between 0 and the first zero of the
    # unobscured pattern (obscuration only moves it inwards).
    v_half = optimize.brentq(half_max, 1e-9, _AIRY_FIRST_ZERO, xtol=1e-12)
    return float(2.0 * v_half / np.pi)


def airy_2d(
    shape: Any,
    fwhm: float | None = None,
    lambda_over_d: float | None = None,
    center: tuple[float, float] | None = None,
    amplitude: float = 1.0,
    obscuration: float = 0.0,
) -> NDArray[np.float64]:
    """Airy diffraction pattern of a circular (optionally obscured) pupil.

    Exactly one of ``fwhm`` or ``lambda_over_d`` must be given.

    With ``obscuration = 0`` the first dark ring falls at
    ``1.220 lambda / D`` and the FWHM at ``1.029 lambda / D`` — both are
    asserted by the test suite.

    Parameters
    ----------
    shape : int or tuple of int
        Output shape.
    fwhm : float, optional
        Desired FWHM in pixels. Converted internally to ``lambda / D`` using
        the exact factor for the requested ``obscuration``.
    lambda_over_d : float, optional
        Diffraction scale ``lambda / D`` expressed in pixels.
    center : tuple of float, optional
        ``(cy, cx)``.
    amplitude : float, default 1.0
        Peak value.
    obscuration : float, default 0.0
        Linear central obscuration ratio (secondary/primary diameter). Keck is
        about 0.14, VLT about 0.14, HST about 0.33. Must be in ``[0, 1)``.

    Returns
    -------
    ndarray
        ``(ny, nx)`` float64 image.
    """
    ny, nx = _as_shape(shape)
    cy, cx = _check_center(center, (ny, nx))

    eps = float(obscuration)
    if not 0.0 <= eps < 1.0:
        raise ValueError(f"obscuration must be in [0, 1); got {eps!r}.")

    if (fwhm is None) == (lambda_over_d is None):
        raise ValueError(
            "airy_2d requires exactly one of fwhm or lambda_over_d "
            f"(got fwhm={fwhm!r}, lambda_over_d={lambda_over_d!r})."
        )

    if lambda_over_d is None:
        lod = _check_positive(fwhm, "fwhm") / _airy_fwhm_factor(eps)
    else:
        lod = _check_positive(lambda_over_d, "lambda_over_d")

    r = _radial_grid((ny, nx), (cy, cx))
    v = np.pi * r / lod
    return float(amplitude) * _airy_amplitude(v, eps) ** 2


def fwhm_lambda_over_d(
    wavelength_m: float, diameter_m: float, pixel_scale: float
) -> float:
    """Diffraction FWHM in pixels for a circular unobscured aperture.

    ``FWHM = 1.0290 * (lambda / D) [arcsec] / pixel_scale``

    Parameters
    ----------
    wavelength_m : float
        Observing wavelength in metres (e.g. ``2.12e-6`` for Ks band).
    diameter_m : float
        Telescope primary diameter in metres (Keck: 10.0, VLT: 8.2).
    pixel_scale : float
        Detector plate scale in arcsec per pixel (NIRC2 narrow: 0.009942).

    Returns
    -------
    float
        FWHM in pixels.

    Examples
    --------
    Keck/NIRC2 narrow camera in Ks band:

    >>> round(fwhm_lambda_over_d(2.12e-6, 10.0, 0.009942), 2)
    4.53
    """
    lam = _check_positive(wavelength_m, "wavelength_m")
    diam = _check_positive(diameter_m, "diameter_m")
    scale = _check_positive(pixel_scale, "pixel_scale")

    lod_arcsec = (lam / diam) * 206264.80624709636
    return _airy_fwhm_factor(0.0) * lod_arcsec / scale


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def _gaussian_model(
    params: NDArray[np.float64],
    yy: NDArray[np.float64],
    xx: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Elliptical Gaussian plus constant offset, evaluated on given coords."""
    amp, y0, x0, sig_x, sig_y, theta, offset = params
    dy = yy - y0
    dx = xx - x0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    return amp * np.exp(-0.5 * ((x_rot / sig_x) ** 2 + (y_rot / sig_y) ** 2)) + offset


def fit_gaussian_psf(
    frame: ArrayLike,
    center_guess: tuple[float, float] | None = None,
    box: int = 15,
) -> dict[str, Any]:
    """Fit an elliptical 2D Gaussian to the core of a PSF.

    Uses :func:`scipy.optimize.least_squares` with a ``soft_l1`` loss, which
    keeps the fit anchored on the core when the wings are contaminated by
    speckles or a nearby source. NaN pixels are excluded from the residual
    vector rather than propagated.

    Parameters
    ----------
    frame : array_like
        2D image containing the source.
    center_guess : tuple of float, optional
        ``(y, x)`` starting point. Defaults to the brightest finite pixel.
    box : int, default 15
        Side of the square fitting region, in pixels. Keep it a few times the
        FWHM: too large and the wings and background dominate the fit.

    Returns
    -------
    dict
        ``{'y', 'x', 'fwhm_x', 'fwhm_y', 'fwhm', 'theta', 'amplitude',
        'offset', 'success'}``. Positions are in **full-frame** coordinates,
        ``fwhm`` is the geometric mean of the two axes, and ``theta`` is in
        degrees within ``[0, 180)``.

    Notes
    -----
    The geometric mean is the right average here: it is what preserves the
    area of the elliptical half-maximum contour, so the equivalent circular
    aperture keeps the same enclosed area.
    """
    arr = _as_frame(frame)
    ny, nx = arr.shape

    box_i = int(box)
    if box_i < 3:
        raise ValueError(f"box must be at least 3 pixels; got {box!r}.")

    if not np.any(np.isfinite(arr)):
        raise ValueError("frame contains no finite pixel; cannot fit a PSF.")

    if center_guess is None:
        gy, gx = np.unravel_index(np.nanargmax(arr), arr.shape)
        gy, gx = float(gy), float(gx)
    else:
        if len(center_guess) != 2:
            raise ValueError(f"center_guess must be a (y, x) pair; got {center_guess!r}.")
        gy, gx = float(center_guess[0]), float(center_guess[1])

    half = box_i // 2
    y0s = max(0, int(round(gy)) - half)
    y1s = min(ny, int(round(gy)) + half + 1)
    x0s = max(0, int(round(gx)) - half)
    x1s = min(nx, int(round(gx)) + half + 1)
    sub = arr[y0s:y1s, x0s:x1s]

    if sub.size < 9:
        raise ValueError(
            f"fitting box is too small after clipping to the frame "
            f"(got {sub.shape}); move center_guess away from the border "
            f"or reduce box."
        )

    yy, xx = np.mgrid[y0s:y1s, x0s:x1s].astype(np.float64)
    finite = np.isfinite(sub)
    if finite.sum() < 7:
        raise ValueError(
            f"only {int(finite.sum())} finite pixels in the fitting box; "
            "at least 7 are needed for a 7-parameter fit."
        )

    sub_f = sub[finite]
    yy_f = yy[finite]
    xx_f = xx[finite]

    # Initial guess from image moments on the background-subtracted core.
    offset0 = float(np.nanmedian(sub))
    amp0 = float(np.nanmax(sub) - offset0)
    if amp0 <= 0:
        amp0 = float(np.nanmax(np.abs(sub))) or 1.0

    wts = np.clip(sub_f - offset0, 0.0, None)
    wsum = wts.sum()
    if wsum > 0:
        y_c = float((wts * yy_f).sum() / wsum)
        x_c = float((wts * xx_f).sum() / wsum)
        var_y = float((wts * (yy_f - y_c) ** 2).sum() / wsum)
        var_x = float((wts * (xx_f - x_c) ** 2).sum() / wsum)
        sig0_y = max(np.sqrt(var_y), 0.5)
        sig0_x = max(np.sqrt(var_x), 0.5)
    else:
        y_c, x_c = gy, gx
        sig0_y = sig0_x = max(box_i / 6.0, 0.5)

    p0 = np.array([amp0, y_c, x_c, sig0_x, sig0_y, 0.0, offset0], dtype=np.float64)
    lo = np.array([0.0, y0s - 1.0, x0s - 1.0, 0.2, 0.2, -np.pi, -np.inf])
    hi = np.array([np.inf, y1s + 1.0, x1s + 1.0, box_i * 2.0, box_i * 2.0, np.pi, np.inf])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

    def residual(params: NDArray[np.float64]) -> NDArray[np.float64]:
        return _gaussian_model(params, yy_f, xx_f) - sub_f

    try:
        res = optimize.least_squares(
            residual, p0, bounds=(lo, hi), loss="soft_l1", f_scale=max(amp0 * 0.1, 1e-12)
        )
        params = res.x
        success = bool(res.success)
    except (ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
        logger.warning("Gaussian PSF fit failed (%s); returning the initial guess.", exc)
        params = p0
        success = False

    amp, y_fit, x_fit, sig_x, sig_y, theta, offset = params
    fwhm_x = float(_SIGMA_TO_FWHM * abs(sig_x))
    fwhm_y = float(_SIGMA_TO_FWHM * abs(sig_y))

    return {
        "y": float(y_fit),
        "x": float(x_fit),
        "fwhm_x": fwhm_x,
        "fwhm_y": fwhm_y,
        "fwhm": float(np.sqrt(fwhm_x * fwhm_y)),
        "theta": float(np.rad2deg(theta) % 180.0),
        "amplitude": float(amp),
        "offset": float(offset),
        "success": success,
    }


def normalize_psf(
    psf: ArrayLike,
    fwhm: float | None = None,
    size: int | None = None,
    threshold: float | None = None,
) -> tuple[NDArray[np.float64], float, float]:
    """Recentre, crop and photometrically normalise an off-axis PSF template.

    This is the calibration step that gives every ``flux`` argument in the
    package a physical meaning. After normalisation, the flux inside a circular
    aperture of **diameter** ``fwhm`` centred on the PSF equals exactly 1, so
    injecting a companion with ``flux = c * star_flux`` produces a companion of
    contrast ``c`` by construction.

    Parameters
    ----------
    psf : array_like
        2D image of the unsaturated, unocculted stellar PSF.
    fwhm : float, optional
        FWHM in pixels. Measured with :func:`fit_gaussian_psf` when omitted.
    size : int, optional
        Side of the output (square) template. The template is cropped or
        zero-padded to this size around the recentred core. Defaults to the
        input size. An odd value is recommended so the core sits on a pixel.
    threshold : float, optional
        After recentring, pixels whose value is below ``threshold`` are set to
        zero. Use it to suppress the noisy wings of a *measured* PSF, which
        would otherwise leak background noise into every injection. Applied
        **before** normalisation, in the units of the input array.

    Returns
    -------
    psf_normalised : ndarray
        The recentred, cropped template with unit aperture flux.
    flux_aperture : float
        The aperture flux of the *input* PSF, i.e. the divisor that was
        applied. This is the number you need if you later want to convert a
        normalised flux back to detector counts.
    fwhm_measured : float
        The FWHM actually used, whether supplied or measured.

    Raises
    ------
    ValueError
        If the aperture flux is not strictly positive, which means the PSF is
        not where the function thinks it is.
    """
    arr = _as_frame(psf, "psf")

    fit = fit_gaussian_psf(arr, box=min(int(min(arr.shape)), 21))
    if fwhm is None:
        fwhm_used = float(fit["fwhm"])
        if not np.isfinite(fwhm_used) or fwhm_used <= 0:
            raise ValueError(
                "could not measure a valid FWHM from the PSF "
                f"(fit returned {fit['fwhm']!r}); pass fwhm explicitly."
            )
        if not fit["success"]:
            logger.warning(
                "PSF Gaussian fit did not converge; the measured FWHM "
                "(%.3f px) may be unreliable.",
                fwhm_used,
            )
    else:
        fwhm_used = _check_positive(fwhm, "fwhm")

    # Recentre so the core sits exactly on the frame centre.
    cy, cx = frame_center(arr.shape)
    dy, dx = cy - fit["y"], cx - fit["x"]
    if abs(dy) > 0.5 * arr.shape[0] or abs(dx) > 0.5 * arr.shape[1]:
        raise ValueError(
            f"the fitted PSF centre ({fit['y']:.2f}, {fit['x']:.2f}) is more "
            f"than half a frame away from the frame centre ({cy:.2f}, {cx:.2f}); "
            "the template is probably not a PSF or is badly cropped."
        )
    if abs(dy) > 1e-3 or abs(dx) > 1e-3:
        arr = frame_shift(arr, dy, dx, order=3, mode="constant", cval=0.0)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if size is not None:
        size_i = int(size)
        if size_i < 3:
            raise ValueError(f"size must be at least 3 pixels; got {size!r}.")
        arr = pad_or_crop(arr, size_i)

    if threshold is not None:
        arr = np.where(arr < float(threshold), 0.0, arr)

    ccy, ccx = frame_center(arr.shape)
    flux_aperture = _aperture_sum(arr, ccy, ccx, fwhm_used / 2.0)
    if not np.isfinite(flux_aperture) or flux_aperture <= 0.0:
        raise ValueError(
            f"the FWHM-diameter aperture flux of the PSF is {flux_aperture!r}, "
            "which cannot be used for normalisation. Check that the PSF is "
            "positive, centred and not entirely masked."
        )

    logger.debug(
        "normalize_psf: fwhm=%.3f px, aperture flux=%.6g, output shape=%s",
        fwhm_used,
        flux_aperture,
        arr.shape,
    )
    return arr / flux_aperture, float(flux_aperture), float(fwhm_used)


def create_synthetic_psf(
    size: int,
    fwhm: float,
    model: str = "airy",
    obscuration: float = 0.14,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Build a normalised synthetic PSF template.

    Convenience wrapper that generates one of the analytic models and passes it
    through :func:`normalize_psf`, so the result is directly usable as an
    injection template.

    Parameters
    ----------
    size : int
        Side of the square output.
    fwhm : float
        FWHM in pixels.
    model : {'airy', 'gaussian', 'moffat'}, default 'airy'
        Which analytic profile to use. ``'airy'`` is the right default for a
        diffraction-limited, AO-corrected observation.
    obscuration : float, default 0.14
        Central obscuration, only used by the Airy model (0.14 is typical of
        Keck and the VLT).
    **kwargs
        Extra keyword arguments forwarded to the model (e.g. ``beta`` for
        Moffat, ``theta``/``fwhm_y`` for an elliptical Gaussian).

    Returns
    -------
    ndarray
        ``(size, size)`` template whose FWHM-diameter aperture flux is 1.
    """
    size_i = int(size)
    if size_i < 3:
        raise ValueError(f"size must be at least 3 pixels; got {size!r}.")
    fwhm_v = _check_positive(fwhm, "fwhm")

    model_l = str(model).lower()
    if model_l == "airy":
        raw = airy_2d((size_i, size_i), fwhm=fwhm_v, obscuration=obscuration, **kwargs)
    elif model_l == "gaussian":
        raw = gaussian_2d((size_i, size_i), fwhm=fwhm_v, **kwargs)
    elif model_l == "moffat":
        raw = moffat_2d((size_i, size_i), fwhm=fwhm_v, **kwargs)
    else:
        raise ValueError(
            f"unknown model {model!r}; expected one of 'airy', 'gaussian', 'moffat'."
        )

    normalised, _, _ = normalize_psf(raw, fwhm=fwhm_v)
    return normalised
