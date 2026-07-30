"""Detection statistics and contrast curves for high-contrast imaging.

Why this module is not just ``(flux - mean) / std``
---------------------------------------------------
At a separation of a few resolution elements there are only a handful of
independent noise samples available — at 3 lambda/D you can fit about 18
non-overlapping resolution elements around the star, at 1.5 lambda/D only 9.
Estimating a noise level from 9 samples and then thresholding at "5 sigma" as if
that noise were known exactly is wrong, and it is wrong in the dangerous
direction: it over-reports significance. Mawet et al. (2014) showed the correct
treatment is a two-sample Student t-test, which costs a penalty factor that
diverges as the separation shrinks.

The consequence is concrete: to claim a detection at the same confidence as a
5-sigma Gaussian threshold, you need a signal-to-noise **ratio** of about 5.6 at
10 resolution elements, but about 9 at 5 of them. Reporting the raw ratio as
"sigma" at small separation is the single most common way to publish a
detection that is not there.

Throughput
----------
Every PSF-subtraction algorithm removes part of the companion along with the
star. A contrast curve computed from the residual noise alone, without dividing
by the measured throughput, is optimistic — by a factor of 2 to 10 at small
separation for aggressive KLIP settings. :func:`throughput` measures it the only
honest way: inject a companion of known flux, run *the same reduction with the
same parameters*, and see how much comes back.

References
----------
Mawet, D. et al. 2014, ApJ, 792, 97 (small-sample statistics; equations 9-10)
Jensen-Clem, R. et al. 2018, AJ, 155, 19 (contrast curve conventions)
Bohn, A. J. et al. 2020, A&A, 642, A48 (throughput and injection practice)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage, signal
from scipy import stats

from .core import (
    azimuthal_positions,
    dist_grid,
    frame_center,
    n_resolution_elements,
)
from .inject import companion_position, inject_companions_cube
from .psf import _aperture_weights

__all__ = [
    "aperture_flux",
    "snr_student",
    "snr_map",
    "noise_profile",
    "throughput",
    "contrast_curve",
    "significance_threshold",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Aperture photometry
# --------------------------------------------------------------------------- #
def aperture_flux(
    frame: ArrayLike,
    y: float,
    x: float,
    radius: float,
    method: str = "exact",
) -> float:
    """Sum of the pixels inside a circular aperture.

    Parameters
    ----------
    frame : array_like
        2D image.
    y, x : float
        Aperture centre, in pixel coordinates (sub-pixel positions allowed).
    radius : float
        Aperture radius in pixels. For the detection statistics in this module
        the convention is ``radius = fwhm / 2``, i.e. an aperture one FWHM
        across — one resolution element.
    method : {'exact', 'center'}, default 'exact'
        ``'exact'`` weights boundary pixels by their overlap fraction, computed
        by 16x16 sub-sampling (better than 0.1 % of the aperture area).
        ``'center'`` counts a pixel if and only if its centre falls inside,
        which is noisier by several percent at these small radii.

    Returns
    -------
    float
        The summed flux. NaN pixels are treated as zero, unless more than half
        the aperture is NaN in which case NaN is returned — a partly-masked
        aperture gives a silently low flux, which would read as a non-detection.
    """
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"frame must be 2D; got shape {arr.shape}.")
    if radius <= 0:
        raise ValueError(f"radius must be positive; got {radius!r}.")

    if method == "exact":
        weights = _aperture_weights(arr.shape, float(y), float(x), float(radius))
    elif method == "center":
        yy = np.arange(arr.shape[0], dtype=np.float64) - float(y)
        xx = np.arange(arr.shape[1], dtype=np.float64) - float(x)
        weights = (np.hypot(yy[:, None], xx[None, :]) <= radius).astype(np.float64)
    else:
        raise ValueError(f"unknown method {method!r}; expected 'exact' or 'center'.")

    total_weight = weights.sum()
    if total_weight <= 0:
        return np.nan

    finite = np.isfinite(arr)
    masked_weight = weights[~finite].sum()
    if masked_weight > 0.5 * total_weight:
        return np.nan

    return float(np.sum(np.where(finite, arr, 0.0) * weights))


def _aperture_kernel(radius: float) -> NDArray[np.float64]:
    """Small centred kernel of aperture overlap weights, for convolution."""
    half = int(np.ceil(radius)) + 1
    n = 2 * half + 1
    return _aperture_weights((n, n), float(half), float(half), float(radius))


def _aperture_flux_map(
    frame: NDArray[np.float64], radius: float
) -> NDArray[np.float64]:
    """Aperture flux at *every* pixel, via one convolution.

    Sliding a circular aperture over an image is a convolution with the aperture
    weights, so the whole map costs one FFT rather than one sum per position.
    This is what makes an exact :func:`snr_map` affordable.
    """
    kernel = _aperture_kernel(radius)
    filled = np.where(np.isfinite(frame), frame, 0.0)
    return signal.fftconvolve(filled, kernel, mode="same")


# --------------------------------------------------------------------------- #
# Signal to noise
# --------------------------------------------------------------------------- #
def _comparison_indices(n: int, exclude_adjacent: bool) -> NDArray[np.intp]:
    """Indices of the reference apertures, position 0 being the test aperture."""
    idx = np.arange(1, n, dtype=np.intp)
    if exclude_adjacent and n >= 5:
        idx = idx[(idx != 1) & (idx != n - 1)]
    return idx


def significance_threshold(
    sigma: float, n_apertures: int, exclude_adjacent: bool = True
) -> float:
    """Ratio needed to reach a Gaussian-equivalent ``sigma`` confidence.

    Implements the inverse of Mawet et al. (2014) equation 10: convert the
    desired false-positive fraction to a Student-t quantile with the degrees of
    freedom actually available, then apply the ``sqrt(1 + 1/n2)`` penalty for
    estimating the mean from a finite sample.

    Parameters
    ----------
    sigma : float
        Gaussian-equivalent confidence, e.g. 5.0.
    n_apertures : int
        Total number of resolution elements at that separation, i.e.
        :func:`exoklip.core.n_resolution_elements`.
    exclude_adjacent : bool, default True
        Whether the two apertures flanking the test position are discarded.

    Returns
    -------
    float
        The threshold in units of the residual noise. Always greater than
        ``sigma``, and sharply so below about 3 lambda/D.

    Examples
    --------
    >>> round(significance_threshold(5.0, 60), 2)   # far out: nearly Gaussian
    5.35
    >>> round(significance_threshold(5.0, 10), 2)   # 1.6 lambda/D: much harsher
    12.34
    """
    n2 = len(_comparison_indices(int(n_apertures), exclude_adjacent))
    if n2 < 2:
        return np.inf
    fpf = float(stats.norm.sf(sigma))
    return float(stats.t.ppf(1.0 - fpf, n2 - 1) * np.sqrt(1.0 + 1.0 / n2))


def snr_student(
    frame: ArrayLike,
    y: float,
    x: float,
    fwhm: float,
    center: Sequence[float] | None = None,
    exclude_adjacent: bool = True,
    full_output: bool = False,
) -> float | dict[str, Any]:
    """Small-sample signal-to-noise ratio at one position (Mawet et al. 2014).

    The noise is estimated from the non-overlapping resolution elements at the
    *same separation* — the only samples that share the test position's speckle
    statistics — and the ratio is

    .. math::
        \\mathrm{SNR} = \\frac{x_1 - \\bar{x}_2}
                             {s_2 \\sqrt{1 + 1/n_2}}

    where :math:`x_1` is the aperture flux at the test position, and
    :math:`\\bar{x}_2`, :math:`s_2` and :math:`n_2` are the mean, the *sample*
    standard deviation (``ddof=1``) and the count of the reference apertures.

    Parameters
    ----------
    frame : array_like
        Reduced 2D image.
    y, x : float
        Position to test, in pixels.
    fwhm : float
        Instrumental FWHM. Apertures have diameter ``fwhm``.
    center : sequence of float, optional
        Stellar centre. Defaults to the frame centre.
    exclude_adjacent : bool, default True
        Discard the two apertures flanking the test position. They overlap the
        companion's own wings, so including them biases the noise upward and the
        mean upward — it costs real sensitivity.
    full_output : bool, default False
        Return a dict with the p-value, the Gaussian-equivalent sigma and the
        individual fluxes instead of a bare ratio.

    Returns
    -------
    float or dict
        The SNR, or a dict with keys ``snr``, ``sigma``, ``p_value``,
        ``n_apertures``, ``n_reference``, ``flux``, ``background_mean``,
        ``background_std``, ``radius``.

    Notes
    -----
    ``snr`` is a *ratio*, not a number of Gaussian sigmas. Use ``sigma`` from
    ``full_output`` — which passes through the Student t distribution — whenever
    you want to state a confidence level. The two differ by more than a factor
    of two inside 2 lambda/D.
    """
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"frame must be 2D; got shape {arr.shape}.")
    if fwhm <= 0:
        raise ValueError(f"fwhm must be positive; got {fwhm!r}.")

    ctr = frame_center(arr.shape) if center is None else (float(center[0]), float(center[1]))
    radius = float(np.hypot(float(y) - ctr[0], float(x) - ctr[1]))
    if radius < fwhm / 2.0:
        raise ValueError(
            f"the test position is only {radius:.2f} px from the centre; "
            f"at least fwhm/2 = {fwhm / 2:.2f} px is required for the aperture "
            "geometry to make sense."
        )

    n = n_resolution_elements(radius, fwhm)
    if n < 4:
        logger.warning(
            "Only %d resolution elements at r=%.1f px: the noise cannot be "
            "estimated. Returning NaN.",
            n,
            radius,
        )
        return {"snr": np.nan, "sigma": np.nan, "n_apertures": n} if full_output else np.nan

    # Aperture positions: index 0 is the test position itself, the rest are
    # spread evenly around the same circle.
    pa_test = np.rad2deg(np.arctan2(-(float(x) - ctr[1]), float(y) - ctr[0])) % 360.0
    positions = azimuthal_positions(radius, fwhm, ctr, pa_offset=pa_test)

    fluxes = np.array(
        [aperture_flux(arr, py, px, fwhm / 2.0) for py, px in positions],
        dtype=np.float64,
    )
    if not np.isfinite(fluxes[0]):
        return {"snr": np.nan, "sigma": np.nan, "n_apertures": n} if full_output else np.nan

    idx = _comparison_indices(n, exclude_adjacent)
    reference = fluxes[idx]
    reference = reference[np.isfinite(reference)]
    n2 = reference.size
    if n2 < 2:
        return {"snr": np.nan, "sigma": np.nan, "n_apertures": n} if full_output else np.nan

    bg_mean = float(reference.mean())
    bg_std = float(reference.std(ddof=1))
    if bg_std <= 0:
        return {"snr": np.inf, "sigma": np.inf, "n_apertures": n} if full_output else np.inf

    snr = (float(fluxes[0]) - bg_mean) / (bg_std * np.sqrt(1.0 + 1.0 / n2))

    if not full_output:
        return float(snr)

    p_value = float(stats.t.sf(snr, n2 - 1))
    return {
        "snr": float(snr),
        "sigma": float(stats.norm.isf(p_value)) if p_value > 0 else np.inf,
        "p_value": p_value,
        "n_apertures": int(n),
        "n_reference": int(n2),
        "flux": float(fluxes[0]),
        "background_mean": bg_mean,
        "background_std": bg_std,
        "radius": radius,
    }


def snr_map(
    frame: ArrayLike,
    fwhm: float,
    center: Sequence[float] | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
    exclude_adjacent: bool = True,
    n_jobs: int = 1,
) -> NDArray[np.float64]:
    """Full small-sample SNR map.

    Computed exactly — every pixel gets the Mawet et al. (2014) statistic with
    the reference apertures at its own separation — but made affordable by
    evaluating the aperture flux everywhere with a single convolution and then
    sampling that map at the reference positions.

    Parameters
    ----------
    frame : array_like
        Reduced 2D image.
    fwhm : float
        Instrumental FWHM in pixels.
    center : sequence of float, optional
        Stellar centre.
    r_min, r_max : float, optional
        Radial range. Defaults to ``fwhm`` and the largest fully-sampled radius.
    exclude_adjacent : bool, default True
        See :func:`snr_student`.
    n_jobs : int, default 1
        Threads used across radial bins.

    Returns
    -------
    ndarray
        Same shape as ``frame``, NaN outside the processed range.
    """
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"frame must be 2D; got shape {arr.shape}.")
    if fwhm <= 0:
        raise ValueError(f"fwhm must be positive; got {fwhm!r}.")

    ny, nx = arr.shape
    ctr = frame_center(arr.shape) if center is None else (float(center[0]), float(center[1]))
    r_grid = dist_grid(arr.shape, center=ctr)

    lo = float(fwhm) if r_min is None else float(r_min)
    lo = max(lo, fwhm / 2.0 + 1.0)
    hi = float(min(ctr[0], ctr[1], ny - 1 - ctr[0], nx - 1 - ctr[1]) - fwhm / 2.0)
    if r_max is not None:
        hi = min(hi, float(r_max))
    if hi <= lo:
        raise ValueError(
            f"empty radial range: r_min={lo:.2f} >= r_max={hi:.2f}. The frame "
            f"({ny}x{nx}) may be too small for fwhm={fwhm}."
        )

    flux_map = _aperture_flux_map(arr, fwhm / 2.0)
    out = np.full(arr.shape, np.nan, dtype=np.float64)

    # Group pixels by integer radius: within a bin the aperture count is
    # constant, so the whole bin is one vectorised operation.
    bins = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1)

    def _process(r_int: int) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
        sel = (r_grid >= r_int) & (r_grid < r_int + 1) & (r_grid >= lo) & (r_grid <= hi)
        ys, xs = np.nonzero(sel)
        if ys.size == 0:
            return ys, xs, np.empty(0)

        r_mid = r_int + 0.5
        n = n_resolution_elements(r_mid, fwhm)
        idx = _comparison_indices(n, exclude_adjacent)
        if n < 4 or idx.size < 2:
            return ys, xs, np.full(ys.size, np.nan)

        # Angles of the reference apertures relative to each test pixel.
        dy = ys - ctr[0]
        dx = xs - ctr[1]
        rad = np.hypot(dy, dx)
        theta0 = np.arctan2(dy, dx)
        offsets = 2.0 * np.pi * idx / n  # (n2,)

        theta = theta0[:, None] + offsets[None, :]  # (m, n2)
        samp_y = ctr[0] + rad[:, None] * np.sin(theta)
        samp_x = ctr[1] + rad[:, None] * np.cos(theta)

        ref = ndimage.map_coordinates(
            flux_map, [samp_y.ravel(), samp_x.ravel()], order=1, mode="constant", cval=np.nan
        ).reshape(theta.shape)

        with np.errstate(invalid="ignore"):
            bg_mean = np.nanmean(ref, axis=1)
            bg_std = np.nanstd(ref, axis=1, ddof=1)
            n2 = np.sum(np.isfinite(ref), axis=1)
            signal_flux = flux_map[ys, xs]
            snr = (signal_flux - bg_mean) / (bg_std * np.sqrt(1.0 + 1.0 / np.maximum(n2, 1)))
        snr[(n2 < 2) | ~np.isfinite(bg_std) | (bg_std <= 0)] = np.nan
        return ys, xs, snr

    if n_jobs and n_jobs > 1:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as pool:
            results = list(pool.map(_process, bins))
    else:
        results = [_process(b) for b in bins]

    for ys, xs, snr in results:
        if ys.size:
            out[ys, xs] = snr
    return out


def noise_profile(
    frame: ArrayLike,
    fwhm: float,
    center: Sequence[float] | None = None,
    method: str = "aperture",
    r_min: float | None = None,
    r_max: float | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Radial profile of the residual noise.

    Parameters
    ----------
    frame : array_like
        Reduced image.
    fwhm : float
        Instrumental FWHM.
    center : sequence of float, optional
        Stellar centre.
    method : {'aperture', 'pixel'}, default 'aperture'
        ``'aperture'`` takes the standard deviation of the fluxes in the
        independent resolution elements at each separation. This is the correct
        estimator for point-source detection and the one a contrast curve must
        use.

        ``'pixel'`` takes the standard deviation of individual pixels in the
        annulus. It is provided for comparison only: because neighbouring pixels
        within a resolution element are strongly correlated, it **underestimates
        the point-source noise** by roughly the square root of the number of
        pixels per resolution element — a factor of about 3.5 for a FWHM of 4
        pixels, i.e. a contrast curve about 1.4 magnitudes too optimistic.
    r_min, r_max : float, optional
        Radial range.

    Returns
    -------
    dict
        ``{'radius', 'noise', 'mean', 'n_apertures'}``.
    """
    arr = np.asarray(frame, dtype=np.float64)
    ctr = frame_center(arr.shape) if center is None else (float(center[0]), float(center[1]))
    ny, nx = arr.shape

    lo = float(fwhm) if r_min is None else float(r_min)
    hi = float(min(ctr[0], ctr[1], ny - 1 - ctr[0], nx - 1 - ctr[1]) - fwhm / 2.0)
    if r_max is not None:
        hi = min(hi, float(r_max))
    radii = np.arange(np.ceil(lo), np.floor(hi) + 1, dtype=np.float64)
    if radii.size == 0:
        raise ValueError(f"empty radial range ({lo:.1f}, {hi:.1f}).")

    if method not in ("aperture", "pixel"):
        raise ValueError(f"unknown method {method!r}; expected 'aperture' or 'pixel'.")

    r_grid = dist_grid(arr.shape, center=ctr) if method == "pixel" else None

    noise = np.empty(radii.size)
    means = np.empty(radii.size)
    counts = np.empty(radii.size, dtype=int)

    for i, r in enumerate(radii):
        if method == "aperture":
            positions = azimuthal_positions(float(r), fwhm, ctr)
            vals = np.array(
                [aperture_flux(arr, py, px, fwhm / 2.0) for py, px in positions]
            )
            vals = vals[np.isfinite(vals)]
            counts[i] = vals.size
            noise[i] = vals.std(ddof=1) if vals.size > 1 else np.nan
            means[i] = vals.mean() if vals.size else np.nan
        else:
            sel = (r_grid >= r - 0.5) & (r_grid < r + 0.5)
            vals = arr[sel]
            vals = vals[np.isfinite(vals)]
            counts[i] = vals.size
            noise[i] = vals.std(ddof=1) if vals.size > 1 else np.nan
            means[i] = vals.mean() if vals.size else np.nan

    return {"radius": radii, "noise": noise, "mean": means, "n_apertures": counts}


# --------------------------------------------------------------------------- #
# Throughput and contrast
# --------------------------------------------------------------------------- #
def throughput(
    cube: ArrayLike,
    angles: ArrayLike,
    psf: ArrayLike,
    fwhm: float,
    radii: Sequence[float],
    reduction_fn: Callable[..., NDArray[np.float64]],
    n_branches: int = 3,
    injection_flux: float | None = None,
    injection_contrast: float = 1e-3,
    star_flux: float = 1.0,
    center: Sequence[float] | None = None,
    **red_kwargs: Any,
) -> dict[str, NDArray[np.float64]]:
    """Fraction of a companion's flux that survives the reduction.

    For each radius and each azimuthal branch: inject one companion of known
    flux, re-run ``reduction_fn`` with **identical** parameters, subtract the
    reduction of the un-injected cube, and measure what is left in a
    FWHM-diameter aperture at the expected position.

    Branches are injected one at a time. Injecting several at once is faster but
    wrong: their KL models contaminate one another and the throughput comes out
    biased.

    Parameters
    ----------
    cube, angles : array_like
        The science sequence and its parallactic angles.
    psf : array_like
        Normalised PSF template (unit FWHM-aperture flux).
    fwhm : float
        Instrumental FWHM in pixels.
    radii : sequence of float
        Separations at which to measure, in pixels.
    reduction_fn : callable
        ``reduction_fn(cube, angles, **red_kwargs) -> image``. Typically
        ``functools.partial(pca_adi, fwhm=fwhm, n_modes=K)``.
    n_branches : int, default 3
        Azimuthal positions per radius, evenly spread. More branches average
        over the local speckle field at the cost of proportionally more
        reductions.
    injection_flux : float, optional
        Absolute flux to inject. Defaults to ``injection_contrast * star_flux``.
    injection_contrast : float, default 1e-3
        Used when ``injection_flux`` is None. Should be bright enough to
        dominate the local noise but faint enough to stay in the linear regime.
    star_flux : float, default 1.0
        Stellar FWHM-aperture flux, used with ``injection_contrast``.
    center : sequence of float, optional
        Stellar centre.
    **red_kwargs
        Forwarded to ``reduction_fn``.

    Returns
    -------
    dict
        ``{'radius', 'throughput', 'throughput_std', 'n_branches'}``.
    """
    arr = np.asarray(cube, dtype=np.float64)
    ang = np.asarray(angles, dtype=np.float64).ravel()
    template = np.asarray(psf, dtype=np.float64)
    radii_arr = np.asarray(radii, dtype=np.float64).ravel()
    if radii_arr.size == 0:
        raise ValueError("radii is empty.")
    if n_branches < 1:
        raise ValueError(f"n_branches must be at least 1; got {n_branches}.")

    ctr = frame_center(arr.shape) if center is None else (float(center[0]), float(center[1]))
    flux = (
        float(injection_flux)
        if injection_flux is not None
        else float(injection_contrast) * float(star_flux)
    )
    if flux <= 0:
        raise ValueError(f"the injection flux must be positive; got {flux!r}.")

    reference_image = np.asarray(reduction_fn(arr, ang, **red_kwargs), dtype=np.float64)

    means = np.empty(radii_arr.size)
    stds = np.empty(radii_arr.size)

    for i, r in enumerate(radii_arr):
        recovered = []
        for b in range(n_branches):
            pa = 360.0 * b / n_branches
            injected = inject_companions_cube(
                arr, template, ang, radius=float(r), pa=pa, flux=flux, center=ctr
            )
            image = np.asarray(reduction_fn(injected, ang, **red_kwargs), dtype=np.float64)
            residual = image - reference_image
            py, px = companion_position(float(r), pa, ctr)
            measured = aperture_flux(residual, py, px, fwhm / 2.0)
            if np.isfinite(measured):
                recovered.append(measured / flux)

        if recovered:
            means[i] = float(np.mean(recovered))
            stds[i] = float(np.std(recovered, ddof=1)) if len(recovered) > 1 else 0.0
        else:
            means[i] = np.nan
            stds[i] = np.nan
        logger.debug("throughput at r=%.1f px: %.3f", r, means[i])

    # A throughput above 1 or below 0 is unphysical and signals a bug upstream
    # (usually a sign error between injection and measurement).
    bad = np.isfinite(means) & ((means > 1.5) | (means < -0.1))
    if np.any(bad):
        logger.warning(
            "Throughput outside [0, 1.5] at radii %s: check that the injection "
            "and derotation sign conventions agree.",
            radii_arr[bad].tolist(),
        )

    return {
        "radius": radii_arr,
        "throughput": means,
        "throughput_std": stds,
        "n_branches": n_branches,
    }


def contrast_curve(
    cube: ArrayLike,
    angles: ArrayLike,
    psf: ArrayLike,
    fwhm: float,
    star_flux: float,
    reduction_fn: Callable[..., NDArray[np.float64]],
    sigma: float = 5.0,
    n_branches: int = 3,
    radii: Sequence[float] | None = None,
    student: bool = True,
    center: Sequence[float] | None = None,
    throughput_result: dict[str, Any] | None = None,
    **red_kwargs: Any,
) -> dict[str, NDArray[np.float64]]:
    """Throughput-corrected detection limit as a function of separation.

    .. math::
        c(r) = \\frac{\\tau_\\sigma(r)\\, \\mathrm{noise}(r) + \\mathrm{bias}(r)}
                    {T(r)\\, F_\\star}

    with :math:`\\tau_\\sigma` the Student-t threshold from
    :func:`significance_threshold`, :math:`T` the measured throughput and
    :math:`F_\\star` the stellar FWHM-aperture flux.

    Parameters
    ----------
    cube, angles, psf, fwhm : array_like / float
        Science sequence, parallactic angles, normalised PSF template, FWHM.
    star_flux : float
        Stellar flux in a FWHM-diameter aperture, **already corrected** for any
        exposure-time ratio and neutral-density filter between the science
        frames and the calibration image. Getting this wrong shifts the whole
        curve by a constant factor and is the most common error in a published
        contrast curve.
    reduction_fn : callable
        ``reduction_fn(cube, angles, **red_kwargs) -> image``.
    sigma : float, default 5.0
        Gaussian-equivalent confidence level.
    n_branches : int, default 3
        Azimuthal branches for the throughput measurement.
    radii : sequence of float, optional
        Separations in pixels. Defaults to one point per FWHM.
    student : bool, default True
        Apply the small-sample penalty. ``False`` reproduces the naive Gaussian
        curve, which is what most of the pre-2014 literature reports and which
        is optimistic at small separation.
    center : sequence of float, optional
        Stellar centre.
    throughput_result : dict, optional
        A previously computed :func:`throughput` output, to avoid repeating the
        expensive injections.
    **red_kwargs
        Forwarded to ``reduction_fn``.

    Returns
    -------
    dict
        ``radius``, ``contrast``, ``contrast_gaussian``, ``throughput``,
        ``noise``, ``bias``, ``sigma_corr`` (the penalty factor, ``> 1``), and
        ``delta_mag`` (``-2.5 log10(contrast)``).
    """
    arr = np.asarray(cube, dtype=np.float64)
    ang = np.asarray(angles, dtype=np.float64).ravel()
    if star_flux <= 0:
        raise ValueError(f"star_flux must be positive; got {star_flux!r}.")

    ctr = frame_center(arr.shape) if center is None else (float(center[0]), float(center[1]))
    ny, nx = arr.shape[1:]
    hi = float(min(ctr[0], ctr[1], ny - 1 - ctr[0], nx - 1 - ctr[1]) - fwhm)
    if radii is None:
        radii_arr = np.arange(float(fwhm), hi, float(fwhm))
    else:
        radii_arr = np.asarray(radii, dtype=np.float64).ravel()
    if radii_arr.size == 0:
        raise ValueError("no valid radius: the frame is too small for this fwhm.")

    reference_image = np.asarray(reduction_fn(arr, ang, **red_kwargs), dtype=np.float64)
    prof = noise_profile(reference_image, fwhm, center=ctr, method="aperture")
    noise = np.interp(radii_arr, prof["radius"], prof["noise"])
    bias = np.interp(radii_arr, prof["radius"], prof["mean"])

    if throughput_result is None:
        throughput_result = throughput(
            arr,
            ang,
            psf,
            fwhm,
            radii_arr,
            reduction_fn,
            n_branches=n_branches,
            injection_contrast=1e-3,
            star_flux=star_flux,
            center=ctr,
            **red_kwargs,
        )
    tput = np.interp(
        radii_arr, throughput_result["radius"], throughput_result["throughput"]
    )
    tput = np.where(tput > 1e-3, tput, np.nan)

    n_ap = np.array([n_resolution_elements(r, fwhm) for r in radii_arr])
    thresholds = np.array([significance_threshold(sigma, int(n)) for n in n_ap])
    sigma_corr = thresholds / float(sigma)

    used = thresholds if student else np.full_like(thresholds, float(sigma))
    contrast = (used * noise + bias) / (tput * star_flux)
    contrast_gauss = (float(sigma) * noise + bias) / (tput * star_flux)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_mag = -2.5 * np.log10(contrast)

    return {
        "radius": radii_arr,
        "contrast": contrast,
        "contrast_gaussian": contrast_gauss,
        "throughput": tput,
        "noise": noise,
        "bias": bias,
        "sigma_corr": sigma_corr,
        "n_apertures": n_ap,
        "delta_mag": delta_mag,
    }
