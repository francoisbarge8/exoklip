"""Fake companion injection — throughput, contrast curves and NEGFC.

Injecting a *known* source is the only way to calibrate what a PSF-subtraction
algorithm does to a real one: KLIP and cADI both remove part of the planet flux
(self-subtraction and over-subtraction), so every contrast curve, every
throughput measurement and every NEGFC fit is built on top of this module.

Conventions
-----------
* Images are ``(y, x)``, cubes are ``(n_frames, y, x)``, ``float64`` internally.
* Position angle (PA): 0 deg = North = ``+y``, increasing towards East = ``-x``
  (same as :func:`exoklip.core.angle_grid` with ``convention='pa'``), hence

  .. math::
      y = c_y + r\\cos(\\mathrm{PA}), \\qquad x = c_x - r\\sin(\\mathrm{PA})

* ``psf_template`` is assumed **normalised** (unit flux in one FWHM-diameter
  aperture, see :func:`exoklip.psf.normalize_psf`) and **centred** on
  ``exoklip.core.frame_center(psf_template.shape)``. ``flux`` is then directly
  expressed in aperture-flux units, i.e. ``flux = contrast * star_flux``.

The sign convention (read this before touching anything)
--------------------------------------------------------
This is *the* place where a sign error hides, so the rule is derived from the
actual behaviour of :mod:`exoklip.rotation` rather than assumed:

1. :func:`exoklip.rotation.frame_rotate` rotates counter-clockwise and
   **increases** the position angle: a source at PA ``p`` ends up at PA
   ``p + angle``.
2. :func:`exoklip.rotation.cube_derotate` rotates frame ``i`` by
   ``sign * angles[i]`` (default ``sign = -1``).

Therefore a companion sitting at PA ``p_i`` in frame ``i`` is found at
``p_i + sign * angles[i]`` in the derotated cube. Requiring that to equal the
requested ``pa`` gives the injection rule implemented here:

.. math::
    p_i = \\mathrm{pa} - \\mathrm{sign} \\times \\mathrm{angles}[i]

which is exactly "rotate the sky position by ``-sign * angles[i]`` to go back
into the detector frame". With the default ``sign = -1`` this reads
``p_i = pa + angles[i]``. Injection and derotation must always be called with
the *same* ``sign``.

References
----------
Marois et al. 2006, ApJ 641, 556 — Angular Differential Imaging.
Lafreniere et al. 2007, ApJ 660, 770 — throughput calibration by injection.
Lagrange et al. 2010, Science 329, 57; Wertz et al. 2017, A&A 598, A83 —
negative fake companion (NEGFC) technique.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import frame_center
from .rotation import frame_shift

__all__ = [
    "inject_companion",
    "inject_companions_cube",
    "remove_companion",
    "companion_position",
]

logger = logging.getLogger(__name__)

#: Default derotation sign. **Must** stay identical to the default of
#: :func:`exoklip.rotation.cube_derotate`, otherwise every injected companion
#: lands at a mirrored position angle.
DEFAULT_SIGN: float = -1.0


# --------------------------------------------------------------------------- #
# private helpers                                                             #
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
        A new ``(ny, nx)`` float64 array; the input is never mutated.

    Raises
    ------
    ValueError
        If ``frame`` is not 2D or has a zero-length axis.
    """
    arr = np.array(frame, dtype=np.float64, copy=True)
    if arr.ndim != 2:
        raise ValueError(
            f"`{name}` must be a 2D image (y, x); got ndim={arr.ndim} with "
            f"shape {arr.shape}."
        )
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(
            f"`{name}` must have strictly positive dimensions; got shape "
            f"{arr.shape}."
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
        A new ``(n_frames, ny, nx)`` float64 array.

    Raises
    ------
    ValueError
        If ``cube`` is not 3D or has a zero-length axis.
    """
    arr = np.array(cube, dtype=np.float64, copy=True)
    if arr.ndim != 3:
        raise ValueError(
            f"`{name}` must be a 3D cube (n_frames, y, x); got ndim={arr.ndim} "
            f"with shape {arr.shape}."
        )
    if min(arr.shape) < 1:
        raise ValueError(
            f"`{name}` must have strictly positive dimensions; got shape "
            f"{arr.shape}."
        )
    return arr


def _check_center(
    center: Sequence[float] | NDArray[np.float64] | None,
    shape: tuple[int, ...],
) -> tuple[float, float]:
    """Resolve ``center`` to a finite ``(cy, cx)`` pair.

    Parameters
    ----------
    center : sequence of float or None
        ``(cy, cx)`` in pixels, or None to use
        :func:`exoklip.core.frame_center` of the last two axes.
    shape : tuple of int
        Shape the centre refers to (2D or 3D).

    Returns
    -------
    tuple of float
        ``(cy, cx)``.

    Raises
    ------
    ValueError
        If ``center`` is not a finite 2-element sequence.
    """
    if center is None:
        cy, cx = frame_center(shape)
        return float(cy), float(cx)
    arr = np.asarray(center, dtype=np.float64).ravel()
    if arr.size != 2:
        raise ValueError(
            f"`center` must be a (cy, cx) pair; got {arr.size} element(s): "
            f"{center!r}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`center` must be finite; got {center!r}.")
    return float(arr[0]), float(arr[1])


def _check_order(order: int) -> int:
    """Validate a spline interpolation order in ``[0, 5]``.

    Parameters
    ----------
    order : int
        Candidate order.

    Returns
    -------
    int
        The validated order.

    Raises
    ------
    ValueError
        If ``order`` is not an integer in ``[0, 5]``.
    """
    try:
        order_int = int(order)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"`order` must be an int in [0, 5]; got {order!r}.") from exc
    if order_int != order or not 0 <= order_int <= 5:
        raise ValueError(f"`order` must be an int in [0, 5]; got {order!r}.")
    return order_int


def _as_scalar(value: object, name: str) -> float:
    """Validate a finite scalar argument.

    Parameters
    ----------
    value : object
        Candidate scalar.
    name : str
        Name used in the error message.

    Returns
    -------
    float
        The value as a Python float.

    Raises
    ------
    ValueError
        If the value is not a finite 0-dimensional number.
    """
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be a finite scalar; got {value!r}.") from exc
    if arr.ndim != 0 or not np.isfinite(arr):
        raise ValueError(f"`{name}` must be a finite scalar; got {value!r}.")
    return float(arr)


def _prepare_psf(
    psf_template: ArrayLike, order: int, name: str = "psf_template"
) -> NDArray[np.float64]:
    """Validate the PSF stamp and pad it so sub-pixel shifts cannot clip it.

    The stamp is zero-padded by ``max(2, order + 1)`` pixels on every side.
    Symmetric padding moves the stamp centre — in the
    :func:`exoklip.core.frame_center` sense — by exactly ``pad`` along each
    axis, so the centring convention is preserved for both parities.

    Parameters
    ----------
    psf_template : array_like
        2D PSF stamp, centred on ``frame_center(psf_template.shape)``.
    order : int
        Spline order the stamp will be shifted with (sets the padding width).
    name : str, optional
        Name used in the error messages.

    Returns
    -------
    numpy.ndarray
        Padded ``float64`` copy of the stamp.

    Raises
    ------
    ValueError
        If the stamp is not 2D or is empty.

    Notes
    -----
    Non-finite pixels (a masked/cropped template) are set to 0 and a warning is
    logged: they would otherwise be spread over the whole injected stamp by the
    B-spline pre-filter.
    """
    psf = _as_frame(psf_template, name=name)
    bad = ~np.isfinite(psf)
    if bad.any():
        logger.warning(
            "`%s` contains %d non-finite pixel(s); they are treated as 0 for "
            "the injection.", name, int(bad.sum()),
        )
        psf[bad] = 0.0
    if not np.any(psf != 0.0):
        logger.warning(
            "`%s` is identically zero: the injection will not change the data.",
            name,
        )
    pad = max(2, int(order) + 1)
    return np.pad(psf, pad, mode="constant", constant_values=0.0)


def _add_psf(
    canvas: NDArray[np.float64],
    psf_padded: NDArray[np.float64],
    ty: float,
    tx: float,
    amplitude: float,
    order: int,
) -> bool:
    """Add ``amplitude * psf`` to ``canvas`` **in place**, centred on ``(ty, tx)``.

    The offset between the stamp centre and the requested position is split
    into an integer part (a pure array slice, exact and cheap) and a fractional
    part in ``[-0.5, 0.5]`` (a single :func:`exoklip.rotation.frame_shift`).

    Parameters
    ----------
    canvas : numpy.ndarray
        2D destination image, modified in place.
    psf_padded : numpy.ndarray
        2D stamp already returned by :func:`_prepare_psf` (finite, zero-padded).
    ty, tx : float
        Target position of the stamp centre, in ``canvas`` pixel coordinates.
    amplitude : float
        Multiplicative factor applied to the stamp (negative for NEGFC).
    order : int
        Spline order of the sub-pixel shift.

    Returns
    -------
    bool
        True if at least one pixel of the stamp landed inside ``canvas``.

    Notes
    -----
    ``frame_shift`` is called with ``cval=0.0`` (not NaN): a companion must not
    punch NaN holes into the data. NaN already present in ``canvas`` stays NaN,
    since the operation is a plain addition.
    """
    pcy, pcx = frame_center(psf_padded.shape)
    off_y = float(ty) - pcy
    off_x = float(tx) - pcx
    int_y = int(round(off_y))
    int_x = int(round(off_x))
    frac_y = off_y - int_y
    frac_x = off_x - int_x

    if abs(frac_y) > 1e-12 or abs(frac_x) > 1e-12:
        stamp = frame_shift(
            psf_padded, frac_y, frac_x, order=order, mode="constant", cval=0.0
        )
    else:
        stamp = psf_padded

    ny, nx = canvas.shape
    sy, sx = stamp.shape
    dy0, dy1 = max(0, int_y), min(ny, int_y + sy)
    dx0, dx1 = max(0, int_x), min(nx, int_x + sx)
    if dy0 >= dy1 or dx0 >= dx1:
        return False
    canvas[dy0:dy1, dx0:dx1] += amplitude * stamp[
        dy0 - int_y:dy1 - int_y, dx0 - int_x:dx1 - int_x
    ]
    return True


def _broadcast_sources(
    radius: ArrayLike, pa: ArrayLike, flux: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Broadcast ``radius`` / ``pa`` / ``flux`` to a common ``(n_sources,)``.

    Parameters
    ----------
    radius, pa, flux : array_like
        Scalars or 1D sequences describing one companion per entry. A scalar is
        reused for every companion.

    Returns
    -------
    tuple of numpy.ndarray
        ``(radii, pas, fluxes)``, three 1D float64 arrays of equal length.

    Raises
    ------
    ValueError
        If an argument has more than one dimension, if the lengths are
        incompatible, if a value is non-finite, or if a radius is negative.
    """
    arrays = {}
    for name, value in (("radius", radius), ("pa", pa), ("flux", flux)):
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if arr.ndim != 1:
            raise ValueError(
                f"`{name}` must be a scalar or a 1D sequence (one entry per "
                f"companion); got shape {arr.shape} with ndim={arr.ndim}."
            )
        if arr.size == 0:
            raise ValueError(f"`{name}` must not be empty.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"`{name}` must be finite; got {value!r}.")
        arrays[name] = arr

    sizes = {k: v.size for k, v in arrays.items()}
    n_sources = max(sizes.values())
    for name, size in sizes.items():
        if size not in (1, n_sources):
            raise ValueError(
                f"`radius`, `pa` and `flux` must be scalars or have the same "
                f"length; got lengths {sizes} (`{name}` has {size}, expected 1 "
                f"or {n_sources})."
            )
    radii = np.broadcast_to(arrays["radius"], (n_sources,)).astype(np.float64)
    pas = np.broadcast_to(arrays["pa"], (n_sources,)).astype(np.float64)
    fluxes = np.broadcast_to(arrays["flux"], (n_sources,)).astype(np.float64)

    if np.any(radii < 0):
        raise ValueError(
            f"`radius` must be >= 0 pixels; got {radii[radii < 0].tolist()}."
        )
    return radii, pas, fluxes


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #
def companion_position(
    radius: float, pa: float, center: Sequence[float]
) -> tuple[float, float]:
    """Cartesian position ``(y, x)`` of a companion at ``(radius, pa)``.

    Package PA convention (identical to
    :func:`exoklip.core.angle_grid` with ``convention='pa'`` and to
    :func:`exoklip.core.azimuthal_positions`): 0 deg = North = ``+y``,
    increasing towards East = ``-x``.

    .. math::
        y = c_y + r\\cos(\\mathrm{PA}), \\qquad x = c_x - r\\sin(\\mathrm{PA})

    Parameters
    ----------
    radius : float
        Separation from ``center``, in pixels, ``>= 0``.
    pa : float
        Position angle in degrees. Any real value, wrapping is implicit.
    center : sequence of float
        ``(cy, cx)`` in pixels.

    Returns
    -------
    tuple of float
        ``(y, x)`` in pixels, sub-pixel accurate.

    Raises
    ------
    ValueError
        If ``radius`` is negative or non-finite, ``pa`` is non-finite, or
        ``center`` is not a finite 2-element sequence.

    Examples
    --------
    >>> y, x = companion_position(10.0, 0.0, (50.0, 50.0))   # due North
    >>> round(y, 6), round(x, 6)
    (60.0, 50.0)
    >>> y, x = companion_position(10.0, 90.0, (50.0, 50.0))  # due East
    >>> round(y, 6), round(x, 6)
    (50.0, 40.0)
    """
    radius = _as_scalar(radius, "radius")
    if radius < 0:
        raise ValueError(f"`radius` must be >= 0 pixels; got {radius}.")
    pa = _as_scalar(pa, "pa")
    if center is None:
        raise ValueError(
            "`center` is required and must be a 2-element (cy, cx) sequence; "
            "got None."
        )
    cy, cx = _check_center(center, (1, 1))
    pa_rad = math.radians(pa)
    return cy + radius * math.cos(pa_rad), cx - radius * math.sin(pa_rad)


def inject_companion(
    frame: ArrayLike,
    psf_template: ArrayLike,
    radius: float,
    pa: float,
    flux: float,
    center: Sequence[float] | None = None,
    order: int = 3,
) -> NDArray[np.float64]:
    """Add one PSF-shaped fake companion at ``(radius, pa)`` in a single frame.

    ``psf_template`` **must** be normalised so that the flux inside one
    FWHM-diameter aperture equals 1 (see
    :func:`exoklip.psf.normalize_psf`); ``flux`` is then directly an aperture
    flux, so a companion at contrast ``c`` is injected with
    ``flux = c * star_flux``.

    Parameters
    ----------
    frame : array_like
        2D image ``(y, x)``. Never modified.
    psf_template : array_like
        2D normalised PSF stamp, centred on
        ``exoklip.core.frame_center(psf_template.shape)``. It may be smaller or
        larger than ``frame``; the part falling outside is clipped.
    radius : float
        Separation from ``center``, in pixels, ``>= 0``.
    pa : float
        Position angle in degrees (0 = North = ``+y``, increasing towards
        East = ``-x``).
    flux : float
        Aperture flux of the companion, in the units of ``frame``. Negative
        values remove flux (NEGFC).
    center : sequence of float, optional
        Star position ``(cy, cx)``. Defaults to
        :func:`exoklip.core.frame_center`.
    order : int, optional
        Spline order of the sub-pixel positioning, in ``[0, 5]``. Default 3.

    Returns
    -------
    numpy.ndarray
        New ``float64`` frame, same shape as ``frame``, with the companion
        added.

    Raises
    ------
    ValueError
        If ``frame``/``psf_template`` are not 2D, if ``radius`` is negative or
        non-finite, if ``pa``/``flux``/``center`` are invalid, or if ``order``
        is outside ``[0, 5]``.

    Notes
    -----
    The position is reached by an integer array slice plus a single sub-pixel
    :func:`exoklip.rotation.frame_shift` of the stamp (zero-padded beforehand,
    so no flux is clipped by the shift). NaN pixels of ``frame`` stay NaN — the
    injection is a plain addition and never fills masked regions.

    References
    ----------
    Lafreniere et al. 2007, ApJ 660, 770 — fake companions for throughput.

    Examples
    --------
    >>> import numpy as np
    >>> psf = np.zeros((5, 5)); psf[2, 2] = 1.0
    >>> out = inject_companion(np.zeros((41, 41)), psf, 10.0, 90.0, 3.0)
    >>> float(out[20, 10])          # East of the centre (20, 20)
    3.0
    """
    arr = _as_frame(frame)
    order = _check_order(order)
    psf = _prepare_psf(psf_template, order)
    radius = _as_scalar(radius, "radius")
    if radius < 0:
        raise ValueError(f"`radius` must be >= 0 pixels; got {radius}.")
    pa = _as_scalar(pa, "pa")
    flux = _as_scalar(flux, "flux")
    cy, cx = _check_center(center, arr.shape)

    ty, tx = companion_position(radius, pa, (cy, cx))
    if not _add_psf(arr, psf, ty, tx, flux, order):
        logger.warning(
            "Companion at (radius=%.3f, pa=%.3f) maps to (y=%.2f, x=%.2f), "
            "entirely outside the frame of shape %s; nothing was injected.",
            radius, pa, ty, tx, arr.shape,
        )
    return arr


def inject_companions_cube(
    cube: ArrayLike,
    psf_template: ArrayLike,
    angles: ArrayLike,
    radius: ArrayLike,
    pa: ArrayLike,
    flux: ArrayLike,
    center: Sequence[float] | None = None,
    n_branches: int = 1,
    sign: float = DEFAULT_SIGN,
    order: int = 3,
) -> NDArray[np.float64]:
    """Inject fake companions into an ADI sequence at a **fixed sky** position.

    In a pupil-tracking sequence the sky rotates in the detector frame, so a
    companion whose sky position angle is ``pa`` sits at a *different* detector
    position angle in every frame. Since
    :func:`exoklip.rotation.cube_derotate` rotates frame ``i`` by
    ``sign * angles[i]``, and :func:`exoklip.rotation.frame_rotate` *increases*
    the position angle for a positive rotation, the companion must be injected
    at

    .. math::
        \\mathrm{PA}_i = \\mathrm{pa} - \\mathrm{sign} \\times \\mathrm{angles}[i]

    so that it lands exactly at ``pa`` after derotation. **Always call this
    function and** :func:`exoklip.rotation.cube_derotate` **with the same**
    ``sign`` (both default to ``-1``).

    Parameters
    ----------
    cube : array_like
        3D ADI cube ``(n_frames, y, x)``. Never modified.
    psf_template : array_like
        2D normalised PSF stamp (unit flux in one FWHM aperture), centred on
        ``exoklip.core.frame_center(psf_template.shape)``.
    angles : array_like
        Parallactic angles in degrees, shape ``(n_frames,)`` — the very same
        array that will be passed to :func:`exoklip.rotation.cube_derotate`.
    radius : float or array_like
        Separation(s) in pixels, ``>= 0``. Scalar, or 1D of length
        ``n_sources``.
    pa : float or array_like
        Sky position angle(s) in degrees, i.e. the PA the companion must show
        **after** derotation. Scalar, or 1D of length ``n_sources``.
    flux : float or array_like
        Aperture flux/fluxes. Scalar, or 1D of length ``n_sources``. Negative
        values remove flux (see :func:`remove_companion`).
    center : sequence of float, optional
        Star position ``(cy, cx)``, common to all frames. Defaults to
        :func:`exoklip.core.frame_center`.
    n_branches : int, optional
        Number of azimuthal copies per source, regularly spaced: branch ``k``
        is injected at ``pa + k * 360 / n_branches``. Default 1.
    sign : float, optional
        Derotation sign, **must match** the one used by
        :func:`exoklip.rotation.cube_derotate`. Default ``-1.0``.
    order : int, optional
        Spline order of the sub-pixel positioning, in ``[0, 5]``. Default 3.

    Returns
    -------
    numpy.ndarray
        New ``float64`` cube ``(n_frames, y, x)`` with
        ``n_sources * n_branches`` companions added per frame.

    Raises
    ------
    ValueError
        If ``cube`` is not 3D, if ``angles`` is not 1D of length ``n_frames``
        or contains non-finite values, if ``radius``/``pa``/``flux`` cannot be
        broadcast to a common length or are non-finite, if ``n_branches < 1``,
        or if ``sign``/``order``/``center`` are invalid.

    Notes
    -----
    Branches are injected into the *same* output cube. When measuring a
    throughput they should be injected **separately** (one call per branch) so
    that neighbouring branches do not contaminate each other's aperture — see
    :func:`exoklip.metrics.throughput`.

    References
    ----------
    Marois et al. 2006, ApJ 641, 556 — ADI field rotation.
    Lafreniere et al. 2007, ApJ 660, 770 — injection-based throughput.

    Examples
    --------
    >>> import numpy as np
    >>> from exoklip.rotation import cube_derotate, cube_collapse
    >>> psf = np.zeros((7, 7)); psf[3, 3] = 1.0
    >>> cube = np.zeros((3, 61, 61))
    >>> angs = np.array([-20.0, 0.0, 20.0])
    >>> cube = inject_companions_cube(cube, psf, angs, 15.0, 0.0, 1.0)
    >>> img = cube_collapse(cube_derotate(cube, angs), mode='median')
    >>> int(np.nanargmax(img) // 61), int(np.nanargmax(img) % 61)
    (45, 30)
    """
    arr = _as_cube(cube)
    n_frames = arr.shape[0]

    ang = np.asarray(angles, dtype=np.float64).ravel()
    if ang.size != n_frames:
        raise ValueError(
            f"`angles` must have one entry per frame: expected shape "
            f"({n_frames},), got {np.shape(angles)}."
        )
    if not np.all(np.isfinite(ang)):
        raise ValueError("`angles` must be finite; got NaN or inf.")

    order = _check_order(order)
    sign = _as_scalar(sign, "sign")
    psf = _prepare_psf(psf_template, order)
    radii, pas, fluxes = _broadcast_sources(radius, pa, flux)

    try:
        n_branches_int = int(n_branches)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"`n_branches` must be an integer >= 1; got {n_branches!r}."
        ) from exc
    if n_branches_int != n_branches or n_branches_int < 1:
        raise ValueError(
            f"`n_branches` must be an integer >= 1; got {n_branches!r}."
        )

    cy, cx = _check_center(center, arr.shape)
    branch_offsets = np.arange(n_branches_int, dtype=np.float64) * (
        360.0 / n_branches_int
    )

    logger.debug(
        "Injecting %d source(s) x %d branch(es) in %d frames "
        "(sign=%+g, order=%d, center=(%.3f, %.3f)).",
        radii.size, n_branches_int, n_frames, sign, order, cy, cx,
    )

    n_outside = 0
    for i in range(n_frames):
        # Detector-frame PA offset: undo the derotation that will be applied.
        detector_offset = -sign * ang[i]
        for radius_s, pa_s, flux_s in zip(radii, pas, fluxes):
            for offset in branch_offsets:
                pa_i = pa_s + offset + detector_offset
                ty, tx = companion_position(radius_s, pa_i, (cy, cx))
                if not _add_psf(arr[i], psf, ty, tx, flux_s, order):
                    n_outside += 1

    if n_outside:
        logger.warning(
            "%d of %d companion placements fell entirely outside the frame "
            "(shape %s, center=(%.3f, %.3f)); check `radius` vs the field of "
            "view.",
            n_outside,
            n_frames * radii.size * n_branches_int,
            arr.shape[1:],
            cy,
            cx,
        )
    return arr


def remove_companion(
    cube: ArrayLike,
    psf_template: ArrayLike,
    angles: ArrayLike,
    radius: ArrayLike,
    pa: ArrayLike,
    flux: ArrayLike,
    center: Sequence[float] | None = None,
    sign: float = DEFAULT_SIGN,
    order: int = 3,
) -> NDArray[np.float64]:
    """Subtract a companion from an ADI sequence — negative injection (NEGFC).

    Strictly equivalent to :func:`inject_companions_cube` with ``-flux`` and
    ``n_branches=1``. Used by the negative fake companion technique: the triplet
    ``(radius, pa, flux)`` that minimises the residuals inside an aperture
    around the source is its best-fit astrometry/photometry.

    Parameters
    ----------
    cube : array_like
        3D ADI cube ``(n_frames, y, x)``. Never modified.
    psf_template : array_like
        2D normalised PSF stamp (unit flux in one FWHM aperture).
    angles : array_like
        Parallactic angles in degrees, shape ``(n_frames,)``.
    radius : float or array_like
        Separation(s) in pixels, ``>= 0``.
    pa : float or array_like
        Sky position angle(s) in degrees (as seen **after** derotation).
    flux : float or array_like
        Aperture flux to **remove** (pass the positive companion flux; the sign
        flip is done internally).
    center : sequence of float, optional
        Star position ``(cy, cx)``. Defaults to
        :func:`exoklip.core.frame_center`.
    sign : float, optional
        Derotation sign, must match :func:`exoklip.rotation.cube_derotate`.
        Default ``-1.0``.
    order : int, optional
        Spline order of the sub-pixel positioning. Default 3.

    Returns
    -------
    numpy.ndarray
        New ``float64`` cube with the companion removed.

    Raises
    ------
    ValueError
        Same conditions as :func:`inject_companions_cube`.

    References
    ----------
    Lagrange et al. 2010, Science 329, 57; Wertz et al. 2017, A&A 598, A83.

    Examples
    --------
    >>> import numpy as np
    >>> psf = np.zeros((5, 5)); psf[2, 2] = 1.0
    >>> angs = np.zeros(2)
    >>> cube = inject_companions_cube(np.zeros((2, 41, 41)), psf, angs,
    ...                               8.0, 45.0, 5.0)
    >>> clean = remove_companion(cube, psf, angs, 8.0, 45.0, 5.0)
    >>> bool(np.allclose(clean, 0.0, atol=1e-9))
    True
    """
    neg_flux = -np.asarray(flux, dtype=np.float64)
    return inject_companions_cube(
        cube,
        psf_template,
        angles,
        radius,
        pa,
        neg_flux,
        center=center,
        n_branches=1,
        sign=sign,
        order=order,
    )
