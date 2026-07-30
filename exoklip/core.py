"""Geometry primitives, masks and resolution elements.

This is the foundation layer of :mod:`exoklip`: it has **no internal
dependency** and defines the conventions every other module relies on.

Conventions
-----------
* Images are ``(y, x)``, cubes are ``(n_frames, y, x)``.
* The frame centre is ``((ny - 1) / 2, (nx - 1) / 2)`` for *both* parities
  (see :func:`frame_center`).
* Angles are in **degrees** in every public API.
* Position angle (PA): ``0`` deg points to North (``+y``) and increases
  towards East (``-x``), i.e. ``PA = trigonometric_angle - 90 deg``.
  Cartesian offsets of a companion at ``(radius, pa)`` are therefore
  ``dy = +radius * cos(pa)`` and ``dx = -radius * sin(pa)``.

References
----------
Mawet et al. 2014, ApJ 792, 97 -- number of independent resolution elements
at a given separation and the associated small-sample statistics.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "frame_center",
    "dist_grid",
    "angle_grid",
    "get_annulus_mask",
    "get_segment_mask",
    "annulus_indices",
    "mask_circle",
    "cube_crop",
    "pad_or_crop",
    "n_resolution_elements",
    "azimuthal_positions",
]


# --------------------------------------------------------------------------- #
# private helpers                                                             #
# --------------------------------------------------------------------------- #
def _as_shape_2d(shape: Any, argname: str = "shape") -> tuple[int, int]:
    """Return the ``(ny, nx)`` spatial shape from a shape tuple or an array.

    Parameters
    ----------
    shape : tuple of int or numpy.ndarray
        Either a shape tuple/list with at least 2 entries (the last two are
        used, so ``(n_frames, y, x)`` is accepted), or an array whose shape is
        used instead.
    argname : str, optional
        Name used in the error message.

    Returns
    -------
    tuple of int
        ``(ny, nx)``.

    Raises
    ------
    ValueError
        If the shape cannot be interpreted or has fewer than 2 axes.
    """
    if isinstance(shape, np.ndarray):
        shape = shape.shape
    if isinstance(shape, (int, np.integer)):
        raise ValueError(
            f"`{argname}` must be a shape tuple with at least 2 axes, "
            f"got the scalar {int(shape)}. Use e.g. ({int(shape)}, {int(shape)})."
        )
    try:
        dims = tuple(int(s) for s in shape)
    except TypeError as exc:  # not iterable
        raise ValueError(
            f"`{argname}` must be a shape tuple or an ndarray, "
            f"got {type(shape).__name__}."
        ) from exc
    if len(dims) < 2:
        raise ValueError(
            f"`{argname}` must have at least 2 axes (y, x), got {dims} "
            f"with {len(dims)} axis/axes."
        )
    ny, nx = dims[-2], dims[-1]
    if ny < 1 or nx < 1:
        raise ValueError(
            f"`{argname}` must have strictly positive spatial dimensions, "
            f"got (ny, nx) = ({ny}, {nx})."
        )
    return ny, nx


def _check_center(
    center: Sequence[float] | np.ndarray | None, shape2d: tuple[int, int]
) -> tuple[float, float]:
    """Validate/complete a ``(cy, cx)`` centre.

    Parameters
    ----------
    center : sequence of float or None
        ``(cy, cx)`` in pixels, or None to use :func:`frame_center`.
    shape2d : tuple of int
        ``(ny, nx)`` used when ``center`` is None.

    Returns
    -------
    tuple of float
        ``(cy, cx)`` as Python floats.
    """
    if center is None:
        return frame_center(shape2d)
    arr = np.asarray(center, dtype=float).ravel()
    if arr.size != 2:
        raise ValueError(
            f"`center` must be a 2-element sequence (cy, cx), got "
            f"{np.shape(center)} with {arr.size} element(s)."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`center` must be finite, got {tuple(arr)}.")
    return float(arr[0]), float(arr[1])


def _check_radii(r_in: float, r_out: float) -> tuple[float, float]:
    """Validate an annulus radius pair (pixels)."""
    r_in = float(r_in)
    r_out = float(r_out)
    if not (math.isfinite(r_in) and math.isfinite(r_out)):
        raise ValueError(f"`r_in`/`r_out` must be finite, got {r_in}, {r_out}.")
    if r_in < 0:
        raise ValueError(f"`r_in` must be >= 0, got {r_in}.")
    if r_out <= r_in:
        raise ValueError(
            f"`r_out` must be strictly greater than `r_in`, got "
            f"r_in={r_in}, r_out={r_out}."
        )
    return r_in, r_out


def _offset_grids(
    shape2d: tuple[int, int], center: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcastable ``(dy, dx)`` offset grids, in pixels, relative to centre.

    ``dy`` has shape ``(ny, 1)`` and ``dx`` has shape ``(1, nx)`` so that any
    ufunc combining them yields a ``(ny, nx)`` map without a Python loop.
    """
    ny, nx = shape2d
    cy, cx = center
    dy = np.arange(ny, dtype=np.float64)[:, np.newaxis] - cy
    dx = np.arange(nx, dtype=np.float64)[np.newaxis, :] - cx
    return dy, dx


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #
def frame_center(shape: tuple[int, ...] | np.ndarray) -> tuple[float, float]:
    """Geometric centre ``(cy, cx)`` of a frame, in pixel coordinates.

    The convention is ``(n - 1) / 2`` along each spatial axis, for **both**
    parities: ``(n - 1) / 2`` when ``n`` is odd (an exact pixel centre) and
    ``n / 2 - 0.5 == (n - 1) / 2`` when ``n`` is even (a pixel corner).

    Why this and not ``n // 2`` or ``n / 2``: pixel *i* of an axis spans index
    coordinates ``[i - 0.5, i + 0.5]`` and the array spans ``0 .. n - 1``, so
    ``(n - 1) / 2`` is the only value that leaves the sampling grid invariant
    under a 180 deg rotation. It is the centre used by
    ``scipy.ndimage.affine_transform`` in :mod:`exoklip.rotation` and by the
    FFT-based image formation in :mod:`exoklip.simulate`; any other choice
    would inject a half-pixel shift at every derotation and break the
    ``rotate(+theta) . rotate(-theta) == identity`` round trip.

    Parameters
    ----------
    shape : tuple of int or numpy.ndarray
        Shape with at least 2 axes (the last two are used, so a cube shape
        ``(n_frames, y, x)`` is accepted), or an array.

    Returns
    -------
    tuple of float
        ``(cy, cx)``.

    Raises
    ------
    ValueError
        If ``shape`` has fewer than 2 axes or non-positive dimensions.

    Examples
    --------
    >>> frame_center((5, 5))
    (2.0, 2.0)
    >>> frame_center((4, 4))
    (1.5, 1.5)
    """
    ny, nx = _as_shape_2d(shape)
    return (ny - 1) / 2.0, (nx - 1) / 2.0


def dist_grid(
    shape: tuple[int, ...] | np.ndarray,
    center: Sequence[float] | None = None,
) -> np.ndarray:
    """Map of the radial distance to ``center``, in pixels.

    Parameters
    ----------
    shape : tuple of int or numpy.ndarray
        Shape with at least 2 axes (last two used), or an array.
    center : sequence of float, optional
        ``(cy, cx)``. Defaults to :func:`frame_center`.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` float64 array of distances.

    Examples
    --------
    >>> dist_grid((3, 3))[1, 1]
    np.float64(0.0)
    """
    shape2d = _as_shape_2d(shape)
    cy, cx = _check_center(center, shape2d)
    dy, dx = _offset_grids(shape2d, (cy, cx))
    return np.hypot(dy, dx)


def angle_grid(
    shape: tuple[int, ...] | np.ndarray,
    center: Sequence[float] | None = None,
    convention: str = "trig",
) -> np.ndarray:
    """Map of angles in degrees, wrapped to ``[0, 360)``.

    Parameters
    ----------
    shape : tuple of int or numpy.ndarray
        Shape with at least 2 axes (last two used), or an array.
    center : sequence of float, optional
        ``(cy, cx)``. Defaults to :func:`frame_center`.
    convention : {'trig', 'pa'}, optional
        ``'trig'``: 0 deg along ``+x``, increasing counter-clockwise (towards
        ``+y``), i.e. ``atan2(dy, dx)``.
        ``'pa'``: astronomical position angle, 0 deg along ``+y`` (North),
        increasing towards ``-x`` (East), i.e. ``atan2(-dx, dy)``.
        The two are related by ``pa = (trig - 90) mod 360``.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` float64 array in ``[0, 360)``.

    Raises
    ------
    ValueError
        If ``convention`` is not one of ``'trig'`` or ``'pa'``.

    Notes
    -----
    The pixel exactly at the centre has an undefined angle; ``atan2(0, 0)``
    returns 0, so it is assigned 0 deg in both conventions.
    """
    shape2d = _as_shape_2d(shape)
    cy, cx = _check_center(center, shape2d)
    conv = str(convention).lower()
    if conv not in ("trig", "pa"):
        raise ValueError(
            f"`convention` must be 'trig' or 'pa', got {convention!r}."
        )
    dy, dx = _offset_grids(shape2d, (cy, cx))
    # Broadcast to a full 2D map before atan2 (dy is (ny,1), dx is (1,nx)).
    dy2d, dx2d = np.broadcast_arrays(dy, dx)
    if conv == "trig":
        ang = np.degrees(np.arctan2(dy2d, dx2d))
    else:  # 'pa': North (+y) is 0, East (-x) is 90
        ang = np.degrees(np.arctan2(-dx2d, dy2d))
    return np.mod(ang, 360.0)


def get_annulus_mask(
    shape: tuple[int, ...] | np.ndarray,
    r_in: float,
    r_out: float,
    center: Sequence[float] | None = None,
) -> np.ndarray:
    """Boolean mask of an annulus, ``r_in <= r < r_out``.

    Parameters
    ----------
    shape : tuple of int or numpy.ndarray
        Shape with at least 2 axes (last two used), or an array.
    r_in, r_out : float
        Inner (inclusive) and outer (exclusive) radii in pixels,
        ``0 <= r_in < r_out``.
    center : sequence of float, optional
        ``(cy, cx)``. Defaults to :func:`frame_center`.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` boolean array; True inside the annulus.

    Raises
    ------
    ValueError
        If the radii are invalid.
    """
    r_in, r_out = _check_radii(r_in, r_out)
    rad = dist_grid(shape, center=center)
    return (rad >= r_in) & (rad < r_out)


def get_segment_mask(
    shape: tuple[int, ...] | np.ndarray,
    r_in: float,
    r_out: float,
    pa_start: float,
    pa_end: float,
    center: Sequence[float] | None = None,
) -> np.ndarray:
    """Boolean mask of an annulus sector (annulus x position-angle wedge).

    The wedge is swept from ``pa_start`` towards **increasing** position angle
    (i.e. from North towards East) up to ``pa_end``; ``pa_start`` is included
    and ``pa_end`` is excluded. Wrap-around through 0/360 is handled: e.g.
    ``pa_start=350, pa_end=10`` selects a 20 deg wedge straddling North.

    Parameters
    ----------
    shape : tuple of int or numpy.ndarray
        Shape with at least 2 axes (last two used), or an array.
    r_in, r_out : float
        Inner (inclusive) and outer (exclusive) radii in pixels.
    pa_start, pa_end : float
        Position angles in degrees, ``'pa'`` convention (0 = North = ``+y``,
        increasing towards East = ``-x``). Any real value is accepted and
        wrapped internally. If the swept width ``(pa_end - pa_start) mod 360``
        is exactly 0 (e.g. ``0`` and ``360``, or two equal angles), the full
        annulus is returned.
    center : sequence of float, optional
        ``(cy, cx)``. Defaults to :func:`frame_center`.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx)`` boolean array; True inside the sector.

    Raises
    ------
    ValueError
        If the radii or the angles are invalid (non-finite).
    """
    r_in, r_out = _check_radii(r_in, r_out)
    pa_start = float(pa_start)
    pa_end = float(pa_end)
    if not (math.isfinite(pa_start) and math.isfinite(pa_end)):
        raise ValueError(
            f"`pa_start`/`pa_end` must be finite degrees, got "
            f"{pa_start}, {pa_end}."
        )

    ann = get_annulus_mask(shape, r_in, r_out, center=center)
    width = (pa_end - pa_start) % 360.0
    if width == 0.0:
        # Full 360 deg wedge (n_segments == 1, or pa_end == pa_start + 360).
        logger.debug(
            "get_segment_mask: pa_start=%.3f, pa_end=%.3f span a full circle, "
            "returning the whole annulus.", pa_start, pa_end,
        )
        return ann

    pa = angle_grid(shape, center=center, convention="pa")
    # Angular distance travelled from pa_start towards increasing PA; the
    # modulo makes the wrap-around through 0/360 automatic.
    delta = np.mod(pa - pa_start, 360.0)
    return ann & (delta < width)


def annulus_indices(
    shape: tuple[int, ...] | np.ndarray,
    r_in: float,
    r_out: float,
    center: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel indices ``(ys, xs)`` of an annulus, for fast zone extraction.

    Parameters
    ----------
    shape : tuple of int or numpy.ndarray
        Shape with at least 2 axes (last two used), or an array.
    r_in, r_out : float
        Inner (inclusive) and outer (exclusive) radii in pixels.
    center : sequence of float, optional
        ``(cy, cx)``. Defaults to :func:`frame_center`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(ys, xs)``, two 1D integer arrays usable as ``cube[:, ys, xs]``.
    """
    mask = get_annulus_mask(shape, r_in, r_out, center=center)
    ys, xs = np.nonzero(mask)
    return ys, xs


def mask_circle(
    array: np.ndarray,
    radius: float,
    center: Sequence[float] | None = None,
    fillwith: float = np.nan,
    mode: str = "in",
) -> np.ndarray:
    """Mask the central disc (``mode='in'``) or everything outside it.

    Parameters
    ----------
    array : numpy.ndarray
        2D frame ``(y, x)`` or 3D cube ``(n_frames, y, x)``. Never modified.
    radius : float
        Radius in pixels, ``>= 0``.
    center : sequence of float, optional
        ``(cy, cx)``. Defaults to :func:`frame_center` of the last two axes.
    fillwith : float, optional
        Value written in the masked pixels. Default ``numpy.nan``.
    mode : {'in', 'out'}, optional
        ``'in'``: pixels with ``r < radius`` are replaced (central disc
        removed, e.g. to blank a coronagraphic core).
        ``'out'``: pixels with ``r >= radius`` are replaced (only the disc is
        kept).

    Returns
    -------
    numpy.ndarray
        A new float64 array with the same shape as ``array``.

    Raises
    ------
    ValueError
        If ``array`` is not 2D/3D, ``radius`` is negative, or ``mode`` is
        unknown.

    Notes
    -----
    The output is always float64 so that ``fillwith=np.nan`` is representable
    even for integer inputs.
    """
    arr = np.asarray(array)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"`array` must be 2D (y, x) or 3D (n_frames, y, x), got shape "
            f"{arr.shape} with {arr.ndim} dimension(s)."
        )
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError(f"`radius` must be a finite value >= 0, got {radius}.")
    mode_l = str(mode).lower()
    if mode_l not in ("in", "out"):
        raise ValueError(f"`mode` must be 'in' or 'out', got {mode!r}.")

    out = np.array(arr, dtype=np.float64, copy=True)
    rad = dist_grid(arr.shape, center=center)
    mask = rad < radius if mode_l == "in" else rad >= radius
    if out.ndim == 3:
        out[:, mask] = fillwith
    else:
        out[mask] = fillwith
    return out


def cube_crop(
    cube: np.ndarray,
    size: int,
    center: Sequence[float] | None = None,
) -> np.ndarray:
    """Crop a frame or a cube to ``(size, size)`` around ``center``.

    An odd ``size`` is recommended: the source centre then lands exactly on
    the central pixel ``(size - 1) / 2`` of the output.

    Parameters
    ----------
    cube : numpy.ndarray
        2D frame ``(y, x)`` or 3D cube ``(n_frames, y, x)``. Never modified.
    size : int
        Side of the square output, in pixels, ``>= 1``.
    center : sequence of float, optional
        ``(cy, cx)`` of the crop. Defaults to :func:`frame_center`.

    Returns
    -------
    numpy.ndarray
        A new array of shape ``(size, size)`` or ``(n_frames, size, size)``.

    Raises
    ------
    ValueError
        If ``size`` exceeds an available dimension, or if the requested window
        falls (partly) outside the array.

    Notes
    -----
    The window start is ``round(c - (size - 1) / 2)`` along each axis, which
    reduces to an exact integer whenever both the centre and ``size`` share
    the parity convention of :func:`frame_center`.
    """
    arr = np.asarray(cube)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"`cube` must be 2D (y, x) or 3D (n_frames, y, x), got shape "
            f"{arr.shape} with {arr.ndim} dimension(s)."
        )
    size_i = int(size)
    if size_i != size or size_i < 1:
        raise ValueError(f"`size` must be a positive integer, got {size!r}.")
    ny, nx = arr.shape[-2], arr.shape[-1]
    if size_i > ny or size_i > nx:
        raise ValueError(
            f"`size`={size_i} is larger than the available frame dimensions "
            f"(ny, nx) = ({ny}, {nx}); use pad_or_crop to pad instead."
        )
    cy, cx = _check_center(center, (ny, nx))

    y0 = int(round(cy - (size_i - 1) / 2.0))
    x0 = int(round(cx - (size_i - 1) / 2.0))
    if y0 < 0 or x0 < 0 or y0 + size_i > ny or x0 + size_i > nx:
        raise ValueError(
            f"The requested crop of size {size_i} around center "
            f"({cy}, {cx}) spans y[{y0}:{y0 + size_i}], x[{x0}:{x0 + size_i}], "
            f"which falls outside the frame of shape ({ny}, {nx})."
        )
    if arr.ndim == 2:
        out = arr[y0:y0 + size_i, x0:x0 + size_i]
    else:
        out = arr[:, y0:y0 + size_i, x0:x0 + size_i]
    return np.array(out, dtype=np.float64, copy=True)


def pad_or_crop(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize a frame/cube to ``(size, size)`` by centred cropping or padding.

    Larger than ``size`` -> centred crop (see :func:`cube_crop`); smaller ->
    zero padding, the input being placed so that its centre (in the
    :func:`frame_center` sense) coincides with the centre of the output.
    Equal -> a plain copy.

    Parameters
    ----------
    frame : numpy.ndarray
        2D frame ``(y, x)`` or 3D cube ``(n_frames, y, x)``. Never modified.
    size : int
        Side of the square output, in pixels, ``>= 1``.

    Returns
    -------
    numpy.ndarray
        A new float64 array of shape ``(size, size)`` or
        ``(n_frames, size, size)``.

    Raises
    ------
    ValueError
        If ``frame`` is not 2D/3D or ``size`` is not a positive integer, or if
        one axis needs cropping while the other needs padding (mixed case is
        handled by padding first, so this only fails on invalid input).
    """
    arr = np.asarray(frame)
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"`frame` must be 2D (y, x) or 3D (n_frames, y, x), got shape "
            f"{arr.shape} with {arr.ndim} dimension(s)."
        )
    size_i = int(size)
    if size_i != size or size_i < 1:
        raise ValueError(f"`size` must be a positive integer, got {size!r}.")

    work = np.array(arr, dtype=np.float64, copy=True)
    ny, nx = work.shape[-2], work.shape[-1]

    # Pad first (independently per axis) so that a mixed pad/crop request is
    # well defined, then crop down to the exact requested size.
    pad_y = max(0, size_i - ny)
    pad_x = max(0, size_i - nx)
    if pad_y or pad_x:
        # Offset that aligns the centres: (size-1)/2 - (n-1)/2 = (size-n)/2.
        off_y = int(round(pad_y / 2.0))
        off_x = int(round(pad_x / 2.0))
        pad_width = [(off_y, pad_y - off_y), (off_x, pad_x - off_x)]
        if work.ndim == 3:
            pad_width = [(0, 0)] + pad_width
        work = np.pad(work, pad_width, mode="constant", constant_values=0.0)

    if work.shape[-1] == size_i and work.shape[-2] == size_i:
        return work
    return cube_crop(work, size_i)


def n_resolution_elements(radius: float, fwhm: float) -> int:
    """Number of independent resolution elements on a circle of given radius.

    ``n = floor(2 * pi * radius / fwhm)``, clipped to a minimum of 1: this is
    the number of non-overlapping FWHM-diameter apertures that fit on the
    circle, i.e. the sample size of the small-sample statistics of
    Mawet et al. 2014 (ApJ 792, 97), their Sect. 3.

    Parameters
    ----------
    radius : float
        Separation from the star, in pixels, ``>= 0``.
    fwhm : float
        Full width at half maximum (aperture diameter), in pixels, ``> 0``.

    Returns
    -------
    int
        Number of independent apertures, ``>= 1``.

    Raises
    ------
    ValueError
        If ``radius < 0`` or ``fwhm <= 0`` or either is not finite.

    Examples
    --------
    >>> n_resolution_elements(10.0, 4.0)
    15
    """
    radius = float(radius)
    fwhm = float(fwhm)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError(f"`radius` must be a finite value >= 0, got {radius}.")
    if not math.isfinite(fwhm) or fwhm <= 0:
        raise ValueError(f"`fwhm` must be a finite value > 0, got {fwhm}.")
    return max(1, int(math.floor(2.0 * math.pi * radius / fwhm)))


def azimuthal_positions(
    radius: float,
    fwhm: float,
    center: Sequence[float] | np.ndarray,
    pa_offset: float = 0.0,
) -> np.ndarray:
    """Positions of the independent apertures on a circle of given radius.

    The ``n = n_resolution_elements(radius, fwhm)`` apertures are regularly
    spaced by ``360 / n`` degrees in position angle, the first one being at
    ``pa_offset`` (Mawet et al. 2014, ApJ 792, 97).

    Positions follow the PA convention of the package: 0 deg = North = ``+y``,
    increasing towards East = ``-x``, i.e.

    .. math::
        y = c_y + r \\cos(\\mathrm{PA}), \\qquad x = c_x - r \\sin(\\mathrm{PA})

    Parameters
    ----------
    radius : float
        Separation from the centre, in pixels, ``>= 0``.
    fwhm : float
        Aperture diameter in pixels, ``> 0``. Only used to set ``n``.
    center : sequence of float
        ``(cy, cx)`` in pixels. Required (there is no shape to infer it from).
    pa_offset : float, optional
        Position angle of the first aperture, in degrees. Default 0 (North).

    Returns
    -------
    numpy.ndarray
        ``(n, 2)`` float64 array of ``(y, x)`` positions, ordered by
        increasing position angle.

    Raises
    ------
    ValueError
        If ``center`` is None/not a 2-element sequence, or if ``radius``/
        ``fwhm``/``pa_offset`` are invalid.

    Examples
    --------
    >>> pos = azimuthal_positions(10.0, 4.0, (0.0, 0.0))
    >>> pos.shape
    (15, 2)
    """
    if center is None:
        raise ValueError(
            "`center` is required for azimuthal_positions and must be a "
            "2-element sequence (cy, cx); got None."
        )
    pa_offset = float(pa_offset)
    if not math.isfinite(pa_offset):
        raise ValueError(f"`pa_offset` must be finite, got {pa_offset}.")
    n = n_resolution_elements(radius, fwhm)  # validates radius and fwhm
    arr = np.asarray(center, dtype=float).ravel()
    if arr.size != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"`center` must be a finite 2-element sequence (cy, cx), got "
            f"{center!r}."
        )
    cy, cx = float(arr[0]), float(arr[1])

    pas = pa_offset + np.arange(n, dtype=np.float64) * (360.0 / n)
    pas_rad = np.radians(pas)
    ys = cy + float(radius) * np.cos(pas_rad)
    xs = cx - float(radius) * np.sin(pas_rad)
    return np.stack((ys, xs), axis=1)
