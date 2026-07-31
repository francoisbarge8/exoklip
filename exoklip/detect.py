"""Source detection and characterisation on a reduced image.

Detection here means one thing precisely: finding local maxima of the
*small-sample SNR map* that survive a stated false-positive threshold, not
finding bright pixels. The difference matters, because a reduced image has
strongly separation-dependent noise — a pixel value that is a solid detection at
40 pixels from the star is unremarkable at 10.

Characterisation uses the negative fake companion technique: rather than
measuring the companion's flux in the reduced image, where PSF subtraction has
already distorted and attenuated it, a companion of *negative* flux is injected
into the raw sequence and its parameters are optimised until the reduction
leaves no residual at that position. This automatically accounts for
self-subtraction instead of trying to correct for it afterwards.

References
----------
Lagrange, A.-M. et al. 2010, Science, 329, 57 (negative companion technique)
Wertz, O. et al. 2017, A&A, 598, A83 (NEGFC error budget)
Mawet, D. et al. 2014, ApJ, 792, 97 (thresholds)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage, optimize

from .core import dist_grid, frame_center
from .inject import companion_position, remove_companion
from .metrics import _student_quantile, aperture_flux, snr_map
from .psf import fit_gaussian_psf

__all__ = ["detect_sources", "negfc_flux", "characterize"]

logger = logging.getLogger(__name__)


def _position_angle(y: float, x: float, center: Sequence[float]) -> float:
    """Position angle in degrees, North (+y) through East (-x)."""
    return float(np.rad2deg(np.arctan2(-(x - center[1]), y - center[0])) % 360.0)


def detect_sources(
    snr_frame: ArrayLike,
    fwhm: float,
    threshold: float = 5.0,
    center: Sequence[float] | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
    exclude_border: bool = True,
    check_fwhm: bool = True,
    image: ArrayLike | None = None,
) -> list[dict[str, Any]]:
    """Find point sources in an SNR map.

    Parameters
    ----------
    snr_frame : array_like
        SNR map, typically from :func:`exoklip.metrics.snr_map`. Passing a raw
        reduced image here is a mistake: its noise varies by orders of magnitude
        with separation, so a single threshold is meaningless.
    fwhm : float
        Instrumental FWHM in pixels.
    threshold : float, default 5.0
        Minimum SNR. Note this is a *ratio*, and at small separation the ratio
        needed for a genuine 5-sigma confidence is much larger. That value is
        reported per candidate as ``threshold_5sigma``, on the same scale as
        ``snr`` so the two can be compared directly. (It is therefore the bare
        Student-t quantile, not
        :func:`exoklip.metrics.significance_threshold`, which is the
        corresponding threshold for the raw aperture-flux ratio used by a
        contrast curve and is larger by ``sqrt(1 + 1/n2)``.)
    center : sequence of float, optional
        Stellar centre.
    r_min, r_max : float, optional
        Radial acceptance range in pixels.
    exclude_border : bool, default True
        Reject candidates within one FWHM of the frame edge.
    check_fwhm : bool, default True
        Reject candidates whose fitted width differs from ``fwhm`` by more than
        a factor of two — cosmic rays are too sharp, residual speckle structures
        and disc features too broad.
    image : array_like, optional
        The reduced image matching ``snr_frame``. When given, the aperture flux
        is measured on it and reported.

    Returns
    -------
    list of dict
        Sorted by decreasing SNR. Keys: ``y``, ``x``, ``radius``, ``pa``,
        ``snr``, ``fwhm_fit``, ``threshold_5sigma``, ``flux_aperture``.
    """
    arr = np.asarray(snr_frame, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"snr_frame must be 2D; got shape {arr.shape}.")
    if fwhm <= 0:
        raise ValueError(f"fwhm must be positive; got {fwhm!r}.")

    ctr = frame_center(arr.shape) if center is None else (float(center[0]), float(center[1]))
    radial = dist_grid(arr.shape, center=ctr)

    finite = np.isfinite(arr)
    # Threshold first, THEN label. Labelling a float image directly — as a naive
    # implementation does — makes every distinct floating-point value its own
    # connected component.
    candidates = finite & (arr >= float(threshold))
    if r_min is not None:
        candidates &= radial >= float(r_min)
    if r_max is not None:
        candidates &= radial <= float(r_max)
    if exclude_border:
        margin = int(np.ceil(fwhm))
        border = np.zeros(arr.shape, dtype=bool)
        border[margin:-margin or None, margin:-margin or None] = True
        candidates &= border

    if not np.any(candidates):
        return []

    footprint = max(int(round(fwhm)), 3)
    filled = np.where(finite, arr, -np.inf)
    local_max = filled >= ndimage.maximum_filter(filled, size=footprint, mode="nearest")
    peaks = candidates & local_max

    labels, n_labels = ndimage.label(peaks)
    if n_labels == 0:
        return []

    reduced = None if image is None else np.asarray(image, dtype=np.float64)
    found: list[dict[str, Any]] = []

    for lab in range(1, n_labels + 1):
        ys, xs = np.nonzero(labels == lab)
        best = int(np.argmax(arr[ys, xs]))
        y0, x0 = float(ys[best]), float(xs[best])
        snr_value = float(arr[int(y0), int(x0)])

        fwhm_fit = np.nan
        source = reduced if reduced is not None else np.where(finite, arr, 0.0)
        try:
            fit = fit_gaussian_psf(source, center_guess=(y0, x0), box=max(int(2 * fwhm) | 1, 7))
            if fit["success"] and np.isfinite(fit["fwhm"]):
                fwhm_fit = float(fit["fwhm"])
                if np.hypot(fit["y"] - y0, fit["x"] - x0) < fwhm:
                    y0, x0 = float(fit["y"]), float(fit["x"])
        except ValueError as exc:  # pragma: no cover - defensive
            logger.debug("Gaussian fit failed for a candidate (%s).", exc)

        if check_fwhm and np.isfinite(fwhm_fit):
            if not (0.5 * fwhm <= fwhm_fit <= 2.0 * fwhm):
                logger.info(
                    "Rejected a candidate at (%.1f, %.1f): fitted FWHM %.2f px is "
                    "inconsistent with the instrumental %.2f px.",
                    y0, x0, fwhm_fit, fwhm,
                )
                continue

        radius = float(np.hypot(y0 - ctr[0], x0 - ctr[1]))
        n_ap = max(int(2 * np.pi * radius / fwhm), 1)
        # `snr` comes from an SNR map, i.e. it is already the t statistic of
        # Mawet et al. (2014) eq. 9, divided by sqrt(1 + 1/n2). The threshold
        # reported next to it must therefore be the bare Student-t quantile:
        # `significance_threshold` carries that factor as well (it is the
        # threshold for the raw aperture-flux ratio a contrast curve uses), and
        # comparing the two would apply the small-sample penalty twice.
        found.append(
            {
                "y": y0,
                "x": x0,
                "radius": radius,
                "pa": _position_angle(y0, x0, ctr),
                "snr": snr_value,
                "fwhm_fit": fwhm_fit,
                "threshold_5sigma": _student_quantile(5.0, n_ap),
                "flux_aperture": (
                    aperture_flux(reduced, y0, x0, fwhm / 2.0)
                    if reduced is not None
                    else np.nan
                ),
            }
        )

    found.sort(key=lambda d: d["snr"], reverse=True)

    # Merge candidates closer than one resolution element.
    kept: list[dict[str, Any]] = []
    for cand in found:
        if all(
            np.hypot(cand["y"] - k["y"], cand["x"] - k["x"]) > fwhm for k in kept
        ):
            kept.append(cand)
    return kept


def negfc_flux(
    cube: ArrayLike,
    angles: ArrayLike,
    psf: ArrayLike,
    fwhm: float,
    radius: float,
    pa: float,
    reduction_fn: Callable[..., NDArray[np.float64]],
    flux_guess: float | None = None,
    center: Sequence[float] | None = None,
    maxiter: int = 120,
    **red_kwargs: Any,
) -> dict[str, Any]:
    """Astrometry and photometry by negative fake companion injection.

    Injects a companion of flux ``-f`` at ``(r, theta)`` into the raw sequence,
    re-runs the reduction, and searches for the ``(r, theta, f)`` that leaves
    the flattest residual inside a 2-FWHM aperture. Because the negative
    companion goes through exactly the same self-subtraction as the real one,
    the recovered flux needs no throughput correction.

    This is expensive: each function evaluation is a full reduction. Expect a
    few hundred reductions for a converged result.

    Parameters
    ----------
    cube, angles : array_like
        Raw sequence and parallactic angles.
    psf : array_like
        Normalised PSF template.
    fwhm : float
        Instrumental FWHM in pixels.
    radius, pa : float
        Starting guess, in pixels and degrees.
    reduction_fn : callable
        ``reduction_fn(cube, angles, **red_kwargs) -> image``.
    flux_guess : float, optional
        Starting flux. Defaults to the aperture flux measured in the reduction
        of the un-modified cube.
    center : sequence of float, optional
        Stellar centre.
    maxiter : int, default 120
        Maximum Nelder-Mead iterations.
    **red_kwargs
        Forwarded to ``reduction_fn``.

    Returns
    -------
    dict
        ``radius``, ``pa``, ``flux``, ``chi2``, ``success``, ``n_eval``.
    """
    arr = np.asarray(cube, dtype=np.float64)
    ang = np.asarray(angles, dtype=np.float64).ravel()
    template = np.asarray(psf, dtype=np.float64)
    ctr = frame_center(arr.shape) if center is None else (float(center[0]), float(center[1]))

    if flux_guess is None:
        baseline = np.asarray(reduction_fn(arr, ang, **red_kwargs), dtype=np.float64)
        y, x = companion_position(radius, pa, ctr)
        measured = aperture_flux(baseline, y, x, fwhm / 2.0)
        flux_guess = float(measured) if np.isfinite(measured) and measured > 0 else 1.0

    calls = {"n": 0}

    def cost(params: NDArray[np.float64]) -> float:
        r, theta, flux = params
        if r <= fwhm / 2.0 or flux < 0:
            return 1e30
        calls["n"] += 1
        cleaned = remove_companion(
            arr, template, ang, radius=float(r), pa=float(theta), flux=float(flux), center=ctr
        )
        image = np.asarray(reduction_fn(cleaned, ang, **red_kwargs), dtype=np.float64)
        y, x = companion_position(float(r), float(theta), ctr)
        yy, xx = np.indices(image.shape)
        window = np.hypot(yy - y, xx - x) <= fwhm
        values = image[window]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 1e30
        return float(np.sum(values**2))

    result = optimize.minimize(
        cost,
        np.array([float(radius), float(pa), float(flux_guess)]),
        method="Nelder-Mead",
        options={"maxiter": int(maxiter), "xatol": 0.01, "fatol": 1e-3},
    )
    r_fit, pa_fit, flux_fit = result.x
    logger.info(
        "NEGFC converged in %d reductions: r=%.2f px, PA=%.2f deg, flux=%.4g",
        calls["n"], r_fit, pa_fit % 360.0, flux_fit,
    )
    return {
        "radius": float(r_fit),
        "pa": float(pa_fit % 360.0),
        "flux": float(flux_fit),
        "chi2": float(result.fun),
        "success": bool(result.success),
        "n_eval": calls["n"],
    }


def characterize(
    cube: ArrayLike,
    angles: ArrayLike,
    psf: ArrayLike,
    fwhm: float,
    star_flux: float,
    reduction_fn: Callable[..., NDArray[np.float64]],
    threshold: float = 5.0,
    do_negfc: bool = True,
    r_min: float | None = None,
    r_max: float | None = None,
    center: Sequence[float] | None = None,
    **red_kwargs: Any,
) -> list[dict[str, Any]]:
    """Detect every candidate and measure it.

    Runs the reduction, builds the SNR map, detects sources, and — unless
    ``do_negfc`` is False — refines each one with :func:`negfc_flux` before
    converting its flux to a contrast.

    Returns
    -------
    list of dict
        The :func:`detect_sources` entries, each augmented with ``contrast``,
        ``delta_mag`` and, when NEGFC ran, the refined ``radius``, ``pa`` and
        ``flux`` under the keys ``negfc_*``.
    """
    arr = np.asarray(cube, dtype=np.float64)
    ang = np.asarray(angles, dtype=np.float64).ravel()
    ctr = frame_center(arr.shape) if center is None else (float(center[0]), float(center[1]))

    image = np.asarray(reduction_fn(arr, ang, **red_kwargs), dtype=np.float64)
    snr = snr_map(image, fwhm, center=ctr, r_min=r_min, r_max=r_max)
    candidates = detect_sources(
        snr, fwhm, threshold=threshold, center=ctr, r_min=r_min, r_max=r_max, image=image
    )
    logger.info("Detected %d candidate(s) above SNR %.1f.", len(candidates), threshold)

    if not do_negfc and candidates:
        logger.warning(
            "characterize(do_negfc=False) cannot correct for self-subtraction. "
            "The reported photometry is under the key 'contrast_uncorrected' and "
            "is a lower limit: with aggressive KLIP settings it can be too faint "
            "by a factor of two or more. Enable NEGFC, or divide by a throughput "
            "measured with exoklip.metrics.throughput at the same settings."
        )

    for cand in candidates:
        # Always reported, always honestly named: this is the aperture flux left
        # in the reduced image, so it has already been eaten by self-subtraction.
        cand["contrast_uncorrected"] = (
            float(cand["flux_aperture"] / star_flux) if star_flux else np.nan
        )
        if do_negfc:
            fit = negfc_flux(
                arr, ang, psf, fwhm, cand["radius"], cand["pa"], reduction_fn,
                flux_guess=cand["flux_aperture"], center=ctr, **red_kwargs,
            )
            cand["negfc_radius"] = fit["radius"]
            cand["negfc_pa"] = fit["pa"]
            cand["negfc_flux"] = fit["flux"]
            # NEGFC subtracts the companion *before* the reduction, so its flux
            # goes through exactly the same self-subtraction as the real one and
            # needs no throughput correction. Only this deserves the plain name.
            cand["contrast"] = (
                float(fit["flux"] / star_flux) if star_flux else np.nan
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                cand["delta_mag"] = float(-2.5 * np.log10(cand["contrast"]))
    return candidates
