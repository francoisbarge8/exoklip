"""Sub-pixel rotation, derotation, translation and temporal collapse.

All geometric operations rely on :mod:`scipy.ndimage` spline interpolation and
are NaN-safe: non-finite pixels are removed before the spline pre-filter and
restored afterwards, so that a single NaN never smears over its whole
neighbourhood (the classical failure mode of B-spline interpolation).

Conventions
-----------
* Images are ``(y, x)``, cubes are ``(n_frames, y, x)``, ``float64`` internally.
* A **positive** rotation angle is **counter-clockwise** in the display frame
  (``imshow(..., origin='lower')``), i.e. it *increases* the position angle
  defined by :func:`exoklip.core.angle_grid` with ``convention='pa'``
  (0 = North = ``+y``, increasing towards East = ``-x``).
* Angles are in degrees in every public API.

References
----------
Marois et al. 2006, ApJ 641, 556 — Angular Differential Imaging.
Unser 1999, IEEE Signal Process. Mag. 16, 22 — B-spline interpolation.
Gomez Gonzalez et al. 2017, AJ 154, 7 (VIP); Wang et al. 2015 (pyKLIP) —
derotation sign convention.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from .core import frame_center

__all__ = ["frame_rotate", "cube_derotate", "frame_shift", "cube_collapse"]

logger = logging.getLogger(__name__)

#: scipy.ndimage boundary modes accepted by the geometric helpers.
_VALID_MODES: frozenset[str] = frozenset(
    {
        "reflect",
        "grid-mirror",
        "constant",
        "grid-constant",
        "nearest",
        "mirror",
        "grid-wrap",
        "wrap",
    }
)

#: Modes for which output pixels can fall outside the input support and must
#: therefore be filled with ``cval``.
_CONSTANT_MODES: frozenset[str] = frozenset({"constant", "grid-constant"})

_COLLAPSE_MODES: tuple[str, ...] = ("median", "mean", "trimmean", "wmean", "sum")


# --------------------------------------------------------------------------- #
# private helpers
# --------------------------------------------------------------------------- #
def _as_frame(frame: ArrayLike, name: str = "frame") -> NDArray[np.float64]:
    """Validate and copy a 2D input as a float64 array.

    Parameters
    ----------
    frame : array_like
        Candidate 2D image.
    name : str, optional
        Name used in the error message.

    Returns
    -------
    numpy.ndarray
        A new ``float64`` array of shape ``(y, x)``; the input is never mutated.

    Raises
    ------
    ValueError
        If ``frame`` is not 2D.
    """
    arr = np.array(frame, dtype=np.float64, copy=True)
    if arr.ndim != 2:
        raise ValueError(
            f"`{name}` must be a 2D image (y, x); got ndim={arr.ndim} "
            f"with shape {arr.shape}."
        )
    return arr


def _as_cube(cube: ArrayLike, name: str = "cube") -> NDArray[np.float64]:
    """Validate and copy a 3D input as a float64 array.

    Parameters
    ----------
    cube : array_like
        Candidate 3D cube.
    name : str, optional
        Name used in the error message.

    Returns
    -------
    numpy.ndarray
        A new ``float64`` array of shape ``(n_frames, y, x)``.

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


def _as_scalar(value: Any, name: str) -> float:
    """Validate a finite scalar argument.

    Parameters
    ----------
    value : object
        Candidate scalar (Python float, numpy scalar or 0-d array).
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


def _check_center(
    center: tuple[float, float] | Sequence[float] | None,
    shape: tuple[int, ...],
) -> tuple[float, float]:
    """Resolve ``center`` to a ``(cy, cx)`` pair of floats.

    Parameters
    ----------
    center : tuple of float or None
        Rotation/expansion centre ``(cy, cx)`` in pixel units. ``None`` falls
        back to :func:`exoklip.core.frame_center`.
    shape : tuple of int
        Shape of the frame the centre refers to.

    Returns
    -------
    tuple of float
        ``(cy, cx)``.

    Raises
    ------
    ValueError
        If ``center`` does not have exactly two finite elements.
    """
    if center is None:
        cy, cx = frame_center(shape)
        return float(cy), float(cx)
    arr = np.asarray(center, dtype=np.float64).ravel()
    if arr.size != 2:
        raise ValueError(
            f"`center` must be a (cy, cx) pair; got {arr.size} element(s): {center!r}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`center` must be finite; got {center!r}.")
    return float(arr[0]), float(arr[1])


def _check_mode(mode: str) -> str:
    """Validate a scipy.ndimage boundary mode.

    Parameters
    ----------
    mode : str
        Candidate boundary mode.

    Returns
    -------
    str
        The validated mode.

    Raises
    ------
    ValueError
        If the mode is unknown to :mod:`scipy.ndimage`.
    """
    if not isinstance(mode, str) or mode not in _VALID_MODES:
        raise ValueError(
            f"`mode` must be one of {sorted(_VALID_MODES)}; got {mode!r}."
        )
    return mode


def _check_order(order: int) -> int:
    """Validate a spline interpolation order.

    Parameters
    ----------
    order : int
        Candidate spline order.

    Returns
    -------
    int
        The validated order in ``[0, 5]``.

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


def _interpolate_nan_safe(
    data: NDArray[np.float64],
    apply: Callable[[NDArray[np.float64], int, str, float], NDArray[np.float64]],
    order: int,
    mode: str,
    cval: float,
) -> NDArray[np.float64]:
    """Apply a geometric transform while protecting against NaN smearing.

    The B-spline pre-filter is an IIR filter: a single non-finite sample
    contaminates the entire frame. The work-around (identical to the one used by
    VIP and pyKLIP) is to

    1. record the non-finite mask,
    2. set those pixels to 0 before interpolation,
    3. transport the mask with nearest-neighbour interpolation (``order=0``),
       which neither grows nor shrinks the invalid region by more than a pixel,
    4. write NaN back into the transported mask.

    Pixels that fall outside the input support (only possible for the
    ``'constant'`` / ``'grid-constant'`` boundary modes) are filled with
    ``cval``. The data transform itself is always evaluated with a *finite*
    fill value so that ``cval=np.nan`` cannot re-enter the spline pre-filter.

    Parameters
    ----------
    data : numpy.ndarray
        2D input image, already a private float64 copy.
    apply : callable
        ``apply(array, order, mode, cval) -> ndarray`` performing the actual
        geometric transform.
    order : int
        Spline order used for the data (the masks always use ``order=0``).
    mode : str
        Boundary mode.
    cval : float
        Fill value for out-of-support pixels.

    Returns
    -------
    numpy.ndarray
        The transformed image, same shape as ``data``.

    Notes
    -----
    Non-finite inputs (NaN **and** :math:`\\pm\\infty`) are treated as missing
    data and come back as NaN.
    """
    invalid = ~np.isfinite(data)
    filled = np.where(invalid, 0.0, data)

    out = apply(filled, order, mode, 0.0)

    if invalid.any():
        moved = apply(invalid.astype(np.float64), 0, mode, 0.0) > 0.5
        out[moved] = np.nan

    if mode in _CONSTANT_MODES:
        inside = apply(np.ones_like(filled), 0, mode, 0.0)
        out[inside < 0.5] = cval

    return out


def _rotation_matrix(angle: float) -> NDArray[np.float64]:
    """Inverse of the counter-clockwise rotation matrix, in ``(y, x)`` order.

    :func:`scipy.ndimage.affine_transform` maps **output** coordinates to
    **input** coordinates (``input = matrix @ output + offset``), so the matrix
    to hand over is the *inverse* of the geometric rotation one wants to see.

    With ``(y, x)`` index ordering, a counter-clockwise rotation by ``+a`` in
    the display frame reads

    .. math::
        R = \\begin{pmatrix} \\cos a & \\sin a \\\\ -\\sin a & \\cos a\\end{pmatrix}

    (it sends ``(dy, dx) = (1, 0)`` — North — to ``(0, -1)`` — East — for
    ``a = 90``, i.e. it increases the position angle), hence this function
    returns :math:`R^{-1}`.

    Parameters
    ----------
    angle : float
        Rotation angle in degrees, counter-clockwise.

    Returns
    -------
    numpy.ndarray
        ``(2, 2)`` matrix acting on ``(y, x)`` coordinates.
    """
    rad = np.deg2rad(float(angle))
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)


def _n_workers(n_jobs: int) -> int:
    """Translate an ``n_jobs`` specification into a worker count.

    Parameters
    ----------
    n_jobs : int
        Number of workers; ``-1`` means "all available CPUs".

    Returns
    -------
    int
        A worker count ``>= 1``.

    Raises
    ------
    ValueError
        If ``n_jobs`` is 0 or below -1.
    """
    n_jobs = int(n_jobs)
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    if n_jobs < 1:
        raise ValueError(f"`n_jobs` must be >= 1 or -1 (all CPUs); got {n_jobs}.")
    return n_jobs


def _nan_reduce(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a ``np.nan*`` reduction without the "All-NaN slice" warning.

    Parameters
    ----------
    func : callable
        The NaN-aware reduction (``np.nanmedian``, ``np.nanmean``, ...).
    *args, **kwargs
        Forwarded to ``func``.

    Returns
    -------
    Any
        Whatever ``func`` returns; all-NaN slices yield NaN.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(*args, **kwargs)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def frame_rotate(
    frame: ArrayLike,
    angle: float,
    center: tuple[float, float] | None = None,
    order: int = 3,
    mode: str = "constant",
    cval: float = np.nan,
) -> NDArray[np.float64]:
    """Rotate a frame counter-clockwise by ``angle`` degrees about ``center``.

    The rotation is implemented with :func:`scipy.ndimage.affine_transform`
    rather than :func:`scipy.ndimage.rotate` because the latter always rotates
    about the geometric centre of the array and cannot honour an arbitrary
    (sub-pixel) star centre.

    A positive ``angle`` increases the position angle: a source lying North of
    ``center`` (``+y``) moves towards East (``-x``), which matches
    ``exoklip.core.angle_grid(..., convention='pa')``.

    Parameters
    ----------
    frame : array_like
        2D image ``(y, x)``. Not modified.
    angle : float
        Rotation angle in degrees, counter-clockwise (positive = towards
        increasing PA).
    center : tuple of float, optional
        Rotation centre ``(cy, cx)`` in pixels. Defaults to
        :func:`exoklip.core.frame_center`.
    order : int, optional
        Spline interpolation order in ``[0, 5]``. ``order >= 2`` uses the
        B-spline pre-filter (``prefilter=True``); 3 (bicubic) is the default
        compromise between photometric accuracy and ringing.
    mode : {'constant', 'grid-constant', 'nearest', 'reflect', 'mirror', \
'grid-mirror', 'wrap', 'grid-wrap'}, optional
        Boundary handling, see :mod:`scipy.ndimage`.
    cval : float, optional
        Value written where the output falls outside the input support (only
        relevant for the constant modes). Default NaN, so that unobserved
        corners stay flagged and are ignored by the NaN-aware collapses.

    Returns
    -------
    numpy.ndarray
        Rotated image, ``float64``, same shape as ``frame``.

    Raises
    ------
    ValueError
        If ``frame`` is not 2D, or if ``angle``/``center``/``order``/``mode``
        are invalid.

    Notes
    -----
    Non-finite input pixels (NaN, ``inf``) are replaced by 0 before the spline
    pre-filter and restored as NaN afterwards using a nearest-neighbour
    transport of the invalid mask. Without this, the IIR pre-filter would
    propagate a single NaN over the whole frame. Flux is conserved to better
    than ~1 % for smooth sources; cubic splines are not exactly flux
    conserving.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.zeros((21, 21)); img[15, 10] = 1.0   # 5 px North of centre
    >>> out = frame_rotate(img, 90.0)
    >>> int(np.nanargmax(out) // 21), int(np.nanargmax(out) % 21)
    (10, 5)
    """
    arr = _as_frame(frame)
    angle = _as_scalar(angle, "angle")
    order = _check_order(order)
    mode = _check_mode(mode)
    cy, cx = _check_center(center, arr.shape)

    if abs(angle % 360.0) < 1e-12 or abs(angle % 360.0 - 360.0) < 1e-12:
        # Exact no-op: skip interpolation entirely (avoids needless smoothing).
        return arr

    matrix = _rotation_matrix(angle)
    center_vec = np.array([cy, cx], dtype=np.float64)
    offset = center_vec - matrix @ center_vec

    def _apply(
        array: NDArray[np.float64], ordr: int, md: str, fill: float
    ) -> NDArray[np.float64]:
        return ndimage.affine_transform(
            array,
            matrix,
            offset=offset,
            order=ordr,
            mode=md,
            cval=fill,
            prefilter=ordr > 1,
        )

    return _interpolate_nan_safe(arr, _apply, order, mode, float(cval))


def cube_derotate(
    cube: ArrayLike,
    angles: ArrayLike,
    center: tuple[float, float] | None = None,
    order: int = 3,
    sign: float = -1.0,
    n_jobs: int = 1,
) -> NDArray[np.float64]:
    """Derotate an ADI cube: frame ``i`` is rotated by ``sign * angles[i]``.

    In a pupil-tracking (ADI) sequence the field rotates by the parallactic
    angle while the speckles stay fixed in the detector frame. To co-align the
    astrophysical field one must **undo** that rotation, hence the default
    ``sign=-1``: frame ``i`` is rotated by ``-angles[i]`` degrees. This is the
    VIP / pyKLIP convention — with parallactic angles as stored in the headers,
    ``sign=-1`` brings North up and East left in the derotated frames.

    Use ``sign=+1`` for the inverse operation (going from a sky-aligned cube
    back to the detector frame), e.g. when injecting a fake companion at a
    fixed sky position.

    Parameters
    ----------
    cube : array_like
        3D cube ``(n_frames, y, x)``. Not modified.
    angles : array_like
        Parallactic angles in degrees, shape ``(n_frames,)``.
    center : tuple of float, optional
        Rotation centre ``(cy, cx)``, common to all frames. Defaults to
        :func:`exoklip.core.frame_center`.
    order : int, optional
        Spline interpolation order, see :func:`frame_rotate`.
    sign : float, optional
        Multiplier applied to the angles. Default ``-1.0`` (derotation).
    n_jobs : int, optional
        Number of threads used to rotate frames in parallel. ``1`` (default)
        runs serially; ``-1`` uses all CPUs.

    Returns
    -------
    numpy.ndarray
        Derotated cube ``(n_frames, y, x)``, ``float64``. Pixels rotated in
        from outside the field are NaN.

    Raises
    ------
    ValueError
        If ``cube`` is not 3D, if ``angles`` is not 1D of length ``n_frames``,
        if ``angles`` contains non-finite values, or if ``sign`` is not finite.

    References
    ----------
    Marois et al. 2006, ApJ 641, 556.
    Gomez Gonzalez et al. 2017, AJ 154, 7 (VIP).

    Examples
    --------
    >>> import numpy as np
    >>> cube = np.zeros((2, 21, 21)); cube[:, 15, 10] = 1.0
    >>> out = cube_derotate(cube, [0.0, 90.0])   # frame 1 rotated by -90 deg
    >>> int(np.nanargmax(out[1]) // 21), int(np.nanargmax(out[1]) % 21)
    (10, 15)
    """
    arr = _as_cube(cube)
    ang = np.asarray(angles, dtype=np.float64).ravel()
    if ang.size != arr.shape[0]:
        raise ValueError(
            f"`angles` must have one entry per frame: expected shape "
            f"({arr.shape[0]},), got {np.shape(angles)}."
        )
    if not np.all(np.isfinite(ang)):
        raise ValueError("`angles` must be finite; got NaN or inf.")
    sign = _as_scalar(sign, "sign")

    order = _check_order(order)
    cy, cx = _check_center(center, arr.shape)
    rot_angles = sign * ang

    logger.debug(
        "Derotating %d frames (sign=%+g, order=%d, center=(%.3f, %.3f)).",
        arr.shape[0],
        sign,
        order,
        cy,
        cx,
    )

    def _one(i: int) -> NDArray[np.float64]:
        return frame_rotate(
            arr[i], rot_angles[i], center=(cy, cx), order=order, mode="constant",
            cval=np.nan,
        )

    out = np.empty_like(arr)
    workers = _n_workers(n_jobs)
    if workers == 1 or arr.shape[0] == 1:
        for i in range(arr.shape[0]):
            out[i] = _one(i)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, res in enumerate(pool.map(_one, range(arr.shape[0]))):
                out[i] = res
    return out


def frame_shift(
    frame: ArrayLike,
    dy: float,
    dx: float,
    order: int = 3,
    mode: str = "constant",
    cval: float = np.nan,
) -> NDArray[np.float64]:
    """Translate a frame by a (possibly sub-pixel) offset.

    The image content is moved by ``+dy`` rows and ``+dx`` columns: a source at
    ``(y0, x0)`` ends up at ``(y0 + dy, x0 + dx)``.

    Parameters
    ----------
    frame : array_like
        2D image ``(y, x)``. Not modified.
    dy : float
        Shift along the first axis (rows), in pixels.
    dx : float
        Shift along the second axis (columns), in pixels.
    order : int, optional
        Spline interpolation order in ``[0, 5]``. Default 3.
    mode : str, optional
        Boundary mode, see :func:`frame_rotate`.
    cval : float, optional
        Fill value for pixels shifted in from outside the field. Default NaN.

    Returns
    -------
    numpy.ndarray
        Shifted image, ``float64``, same shape as ``frame``.

    Raises
    ------
    ValueError
        If ``frame`` is not 2D or if ``dy``/``dx``/``order``/``mode`` are
        invalid.

    Notes
    -----
    Same NaN protection as :func:`frame_rotate`: non-finite pixels are zeroed
    before the spline pre-filter, transported with ``order=0`` and restored as
    NaN.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.zeros((11, 11)); img[5, 5] = 1.0
    >>> out = frame_shift(img, 2.0, -1.0)
    >>> bool(abs(out[7, 4] - 1.0) < 1e-12)
    True
    """
    arr = _as_frame(frame)
    shift_vec = (_as_scalar(dy, "dy"), _as_scalar(dx, "dx"))
    order = _check_order(order)
    mode = _check_mode(mode)

    if shift_vec == (0.0, 0.0):
        return arr

    def _apply(
        array: NDArray[np.float64], ordr: int, md: str, fill: float
    ) -> NDArray[np.float64]:
        return ndimage.shift(
            array, shift_vec, order=ordr, mode=md, cval=fill, prefilter=ordr > 1
        )

    return _interpolate_nan_safe(arr, _apply, order, mode, float(cval))


def cube_collapse(
    cube: ArrayLike,
    mode: str = "median",
    weights: ArrayLike | None = None,
    trim: float = 0.1,
) -> NDArray[np.float64]:
    """Combine a cube along the temporal axis, ignoring NaN.

    Parameters
    ----------
    cube : array_like
        3D cube ``(n_frames, y, x)``. Not modified.
    mode : {'median', 'mean', 'trimmean', 'wmean', 'sum'}, optional
        Combination rule.

        * ``'median'``  : :func:`numpy.nanmedian` — robust, the ADI default.
        * ``'mean'``    : :func:`numpy.nanmean`.
        * ``'trimmean'``: mean after discarding a fraction ``trim`` of the
          *valid* samples at each end (per pixel).
        * ``'wmean'``   : weighted mean, weights given by ``weights``
          (e.g. the inverse annulus variance of each frame).
        * ``'sum'``     : :func:`numpy.nansum`, NaN where a pixel is NaN in
          every frame.
    weights : array_like, optional
        Only for ``mode='wmean'``. Shape ``(n_frames,)`` (one weight per frame)
        or the full cube shape (per-pixel weights). Must be finite,
        non-negative and not all zero.
    trim : float, optional
        Only for ``mode='trimmean'``. Fraction trimmed at *each* end, in
        ``[0, 0.5)``. Default 0.1.

    Returns
    -------
    numpy.ndarray
        Collapsed image ``(y, x)``, ``float64``. A pixel that is NaN in every
        frame stays NaN for every mode.

    Raises
    ------
    ValueError
        If ``cube`` is not 3D, ``mode`` is unknown, ``trim`` is outside
        ``[0, 0.5)``, or ``weights`` is missing/ill-shaped for ``'wmean'``.

    Notes
    -----
    Every reduction is NaN-aware, because derotated ADI cubes are full of NaN
    corners (see :func:`cube_derotate`). The trimmed mean trims
    ``floor(trim * n_valid)`` samples per side using the number of *finite*
    samples of that pixel, and always keeps at least one sample.

    Examples
    --------
    >>> import numpy as np
    >>> cube = np.array([[[1.0]], [[2.0]], [[np.nan]], [[100.0]]])
    >>> float(cube_collapse(cube, mode='median'))
    2.0
    """
    arr = _as_cube(cube)
    if not isinstance(mode, str) or mode not in _COLLAPSE_MODES:
        raise ValueError(
            f"`mode` must be one of {list(_COLLAPSE_MODES)}; got {mode!r}."
        )
    n_frames = arr.shape[0]

    if mode == "median":
        return _nan_reduce(np.nanmedian, arr, axis=0)

    if mode == "mean":
        return _nan_reduce(np.nanmean, arr, axis=0)

    if mode == "sum":
        valid = np.isfinite(arr)
        out = np.where(valid, arr, 0.0).sum(axis=0)
        out[~valid.any(axis=0)] = np.nan
        return out

    if mode == "trimmean":
        trim = _as_scalar(trim, "trim")
        if not 0.0 <= trim < 0.5:
            raise ValueError(f"`trim` must lie in [0, 0.5); got {trim}.")

        valid = np.isfinite(arr)
        n_valid = valid.sum(axis=0)
        # NaN sorts to the end of each column -> the first n_valid entries are
        # the sorted finite samples.
        srt = np.sort(np.where(valid, arr, np.nan), axis=0)
        k = np.floor(trim * n_valid).astype(np.int64)
        # keep at least one sample
        k = np.minimum(k, np.maximum((n_valid - 1) // 2, 0))
        idx = np.arange(n_frames, dtype=np.int64)[:, None, None]
        keep = (idx >= k[None, :, :]) & (idx < (n_valid - k)[None, :, :])
        count = keep.sum(axis=0)
        total = np.where(keep, np.nan_to_num(srt, nan=0.0), 0.0).sum(axis=0)
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    # mode == 'wmean'
    if weights is None:
        raise ValueError("`weights` is required when mode='wmean'.")
    w = np.asarray(weights, dtype=np.float64)
    if w.shape == (n_frames,):
        w_full = w[:, None, None]
    elif w.shape == arr.shape:
        w_full = w
    else:
        raise ValueError(
            f"`weights` must have shape ({n_frames},) or {arr.shape}; "
            f"got {w.shape}."
        )
    if not np.all(np.isfinite(w_full)):
        raise ValueError("`weights` must be finite; got NaN or inf.")
    if np.any(w_full < 0):
        raise ValueError("`weights` must be non-negative.")
    if not np.any(w_full > 0):
        raise ValueError("`weights` must not be all zero.")

    valid = np.isfinite(arr)
    w_eff = np.where(valid, w_full, 0.0)
    num = (np.where(valid, arr, 0.0) * w_eff).sum(axis=0)
    den = w_eff.sum(axis=0)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
