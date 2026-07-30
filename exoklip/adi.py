"""Angular differential imaging reductions.

Three algorithms, in increasing order of sophistication and of what they cost
you in companion signal:

``median_adi``
    Classical ADI (Marois et al. 2006): subtract the temporal median, derotate,
    combine. One PSF model for the whole sequence. Cheap, robust, and the
    baseline every other result must beat.

``pca_adi(mode='fullframe')``
    A truncated KL model of the stellar PSF built from the whole frame. Better
    than cADI at large separation, but a single set of modes cannot describe a
    field whose speckle statistics vary by orders of magnitude between the core
    and the edge.

``pca_adi(mode='annular')`` — also exported as ``klip_adi``
    KLIP applied zone by zone, with the reference library restricted per zone by
    the field-rotation criterion. This is the reference implementation: the
    model is local, so it adapts to the local speckle field, and the rotation
    threshold protects the companion from being absorbed into its own model.

The self-subtraction trade-off
------------------------------
Every one of these algorithms partly subtracts the companion along with the
star, because the companion is present in the frames used to build the model.
More KL modes means a better stellar model *and* more companion signal removed.
The optimum is a genuine trade-off, which is why :func:`optimize_n_modes`
exists and why any contrast derived from these images is meaningless until it
has been divided by the measured throughput
(:func:`exoklip.metrics.throughput`).

References
----------
Marois, C. et al. 2006, ApJ, 641, 556
Lafreniere, D. et al. 2007, ApJ, 660, 770
Soummer, R., Pueyo, L. & Larkin, J. 2012, ApJL, 755, L28
Amara, A. & Quanz, S. P. 2012, MNRAS, 427, 948
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import mask_circle
from .klip import klip_annular, klip_fullframe
from .rotation import cube_collapse, cube_derotate

__all__ = ["median_adi", "pca_adi", "klip_adi", "optimize_n_modes"]

logger = logging.getLogger(__name__)


def _as_cube(cube: ArrayLike) -> NDArray[np.float64]:
    arr = np.array(cube, dtype=np.float64, copy=True)
    if arr.ndim != 3:
        raise ValueError(
            f"cube must be 3D (n_frames, y, x); got shape {arr.shape} "
            f"with ndim={arr.ndim}."
        )
    if arr.shape[0] < 2:
        raise ValueError(
            f"an ADI reduction needs at least 2 frames; got {arr.shape[0]}."
        )
    return arr


def _check_angles(angles: ArrayLike, n_frames: int) -> NDArray[np.float64]:
    arr = np.asarray(angles, dtype=np.float64).ravel()
    if arr.size != n_frames:
        raise ValueError(
            f"angles has {arr.size} entries but the cube has {n_frames} frames; "
            "there must be exactly one parallactic angle per frame."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("angles contains non-finite values.")
    return arr


def _finalise(
    residuals: NDArray[np.float64],
    angles: NDArray[np.float64],
    center: Sequence[float] | None,
    collapse: str,
    mask_radius: float | None,
    n_jobs: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Derotate a residual cube and collapse it to a final image."""
    derotated = cube_derotate(residuals, angles, center=center, n_jobs=n_jobs)
    image = cube_collapse(derotated, mode=collapse)
    if mask_radius is not None and mask_radius > 0:
        image = mask_circle(image, mask_radius, center=center, fillwith=np.nan, mode="in")
    return image, derotated


def median_adi(
    cube: ArrayLike,
    angles: ArrayLike,
    center: Sequence[float] | None = None,
    collapse: str = "median",
    mask_radius: float | None = None,
    full_output: bool = False,
    n_jobs: int = 1,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], ...]:
    """Classical ADI: subtract the temporal median PSF, derotate, combine.

    Parameters
    ----------
    cube : array_like
        ``(n_frames, y, x)`` sequence in pupil-tracking mode.
    angles : array_like
        ``(n_frames,)`` parallactic angles in degrees.
    center : sequence of float, optional
        ``(cy, cx)`` stellar centre. Defaults to the frame centre.
    collapse : {'median', 'mean', 'trimmean', 'wmean', 'sum'}, default 'median'
        Temporal combination of the derotated residuals.
    mask_radius : float, optional
        Radius in pixels of a central region set to NaN in the output.
    full_output : bool, default False
        Also return the derotated and raw residual cubes.
    n_jobs : int, default 1
        Threads used for derotation.

    Returns
    -------
    ndarray or tuple
        The final image, or ``(image, derotated_residuals, raw_residuals)``.

    Notes
    -----
    The median is taken over frames without any rotation-threshold protection,
    so at small separation — where the companion barely moves between frames —
    it enters its own PSF model and is largely subtracted. That is the known
    weakness this baseline has and that KLIP's ``delta_rot`` addresses.
    """
    arr = _as_cube(cube)
    ang = _check_angles(angles, arr.shape[0])

    psf_model = np.nanmedian(arr, axis=0)
    residuals = arr - psf_model

    image, derotated = _finalise(residuals, ang, center, collapse, mask_radius, n_jobs)
    if full_output:
        return image, derotated, residuals
    return image


def pca_adi(
    cube: ArrayLike,
    angles: ArrayLike,
    fwhm: float,
    n_modes: int | Sequence[int] = 10,
    mode: str = "annular",
    collapse: str = "median",
    delta_rot: float = 1.0,
    asize: float = 4.0,
    n_segments: int | str = 1,
    r_min: float | None = None,
    r_max: float | None = None,
    min_refs: int = 5,
    mask_radius: float | None = None,
    center: Sequence[float] | None = None,
    full_output: bool = False,
    n_jobs: int = 1,
    verbose: bool = False,
) -> Any:
    """KLIP/PCA reduction of an ADI sequence.

    Parameters
    ----------
    cube, angles : array_like
        The sequence and its parallactic angles in degrees.
    fwhm : float
        Instrumental FWHM in pixels; sets the zone widths and the rotation
        threshold.
    n_modes : int or sequence of int, default 10
        Number of KL modes to keep. **A sequence triggers a mode sweep**: the
        eigendecomposition is computed once and reused, and the return value
        becomes a ``{k: image}`` dict. This is much cheaper than calling the
        function once per value.
    mode : {'annular', 'fullframe'}, default 'annular'
        Zonal or global KL model.
    collapse : str, default 'median'
        Temporal combination, see :func:`exoklip.rotation.cube_collapse`.
    delta_rot : float, default 1.0
        Minimum companion displacement, in FWHM, required for a frame to be
        usable as a reference. Ignored in ``fullframe`` mode unless non-zero.
    asize : float, default 4.0
        Annulus width in FWHM (``annular`` mode).
    n_segments : int or 'auto', default 1
        Azimuthal subdivisions per annulus.
    r_min, r_max : float, optional
        Radial range to process, in pixels.
    min_refs : int, default 5
        Minimum size of the reference library before the rotation criterion is
        relaxed.
    mask_radius : float, optional
        Radius of a central NaN mask applied to the output.
    center : sequence of float, optional
        Stellar centre.
    full_output : bool, default False
        Also return the derotated and raw residual cubes.
    n_jobs : int, default 1
        Threads used for the annuli and the derotation.
    verbose : bool, default False
        Log per-zone progress.

    Returns
    -------
    ndarray, dict, or tuple
        Final image; a ``{k: image}`` dict when ``n_modes`` is a sequence; or
        ``(image, derotated_residuals, raw_residuals)`` when ``full_output``.
    """
    arr = _as_cube(cube)
    ang = _check_angles(angles, arr.shape[0])
    if fwhm <= 0:
        raise ValueError(f"fwhm must be positive; got {fwhm!r}.")

    mode_l = str(mode).lower()
    if mode_l == "annular":
        residuals = klip_annular(
            arr,
            ang,
            fwhm,
            n_modes=n_modes,
            asize=asize,
            delta_rot=delta_rot,
            n_segments=n_segments,
            r_min=r_min,
            r_max=r_max,
            min_refs=min_refs,
            center=center,
            verbose=verbose,
            n_jobs=n_jobs,
        )
    elif mode_l == "fullframe":
        residuals = klip_fullframe(
            arr,
            ang,
            fwhm,
            n_modes=n_modes,
            delta_rot=delta_rot,
            mask_radius=mask_radius,
            center=center,
        )
    else:
        raise ValueError(
            f"unknown mode {mode!r}; expected 'annular' or 'fullframe'."
        )

    if isinstance(residuals, dict):
        images: dict[int, NDArray[np.float64]] = {}
        derotated_all: dict[int, NDArray[np.float64]] = {}
        for k, res in residuals.items():
            img, der = _finalise(res, ang, center, collapse, mask_radius, n_jobs)
            images[k] = img
            derotated_all[k] = der
        if full_output:
            return images, derotated_all, residuals
        return images

    image, derotated = _finalise(residuals, ang, center, collapse, mask_radius, n_jobs)
    if full_output:
        return image, derotated, residuals
    return image


def klip_adi(*args: Any, **kwargs: Any) -> Any:
    """KLIP in annular mode — the reference reduction.

    Thin alias for :func:`pca_adi` with ``mode='annular'``, provided because
    "KLIP" in the literature almost always means the zonal variant.
    """
    kwargs.setdefault("mode", "annular")
    return pca_adi(*args, **kwargs)


def optimize_n_modes(
    cube: ArrayLike,
    angles: ArrayLike,
    fwhm: float,
    radius: float,
    pa: float,
    n_modes_grid: Sequence[int],
    center: Sequence[float] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Scan the number of KL modes and report the SNR of a source at (radius, pa).

    The number of modes is the single most consequential free parameter of a
    KLIP reduction, and its optimum is not monotonic: too few modes leave
    speckle residuals, too many start eating the companion. This runs the sweep
    in one eigendecomposition and measures the SNR at a fixed position.

    Parameters
    ----------
    cube, angles : array_like
        The sequence and its parallactic angles.
    fwhm : float
        Instrumental FWHM in pixels.
    radius, pa : float
        Position of the source to optimise for, in pixels and degrees. Use a
        real candidate, or inject a fake one first with
        :func:`exoklip.inject.inject_companions_cube`.
    n_modes_grid : sequence of int
        Values of K to try.
    center : sequence of float, optional
        Stellar centre.
    **kwargs
        Forwarded to :func:`pca_adi`.

    Returns
    -------
    dict
        ``{'n_modes': list, 'snr': list, 'best': int, 'images': dict}``.
    """
    from .core import frame_center
    from .inject import companion_position
    from .metrics import snr_student  # local import: metrics depends on adi

    grid = [int(k) for k in n_modes_grid]
    if not grid:
        raise ValueError("n_modes_grid is empty.")

    images = pca_adi(cube, angles, fwhm, n_modes=grid, center=center, **kwargs)
    if not isinstance(images, dict):  # single value passed through
        images = {grid[0]: images}

    ctr = frame_center(np.asarray(cube).shape) if center is None else center
    y, x = companion_position(radius, pa, ctr)

    snrs: list[float] = []
    for k in grid:
        try:
            snrs.append(float(snr_student(images[k], y, x, fwhm, center=ctr)))
        except (ValueError, ZeroDivisionError) as exc:
            logger.warning("SNR could not be computed for K=%d (%s).", k, exc)
            snrs.append(np.nan)

    finite = [s for s in snrs if np.isfinite(s)]
    best = grid[int(np.nanargmax(snrs))] if finite else grid[0]
    logger.info("Best number of KL modes: %d (SNR = %.2f)", best, max(finite, default=np.nan))

    return {"n_modes": grid, "snr": snrs, "best": best, "images": images}
