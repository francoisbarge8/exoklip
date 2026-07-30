"""Karhunen-Loeve Image Projection (KLIP) -- the core of :mod:`exoklip`.

Implements the truncated Karhunen-Loeve expansion of a set of reference PSFs
and the projection/subtraction of a science frame onto it, following

    Soummer, Pueyo & Larkin 2012, ApJL 755, L28
    ("Detection and Characterization of Exoplanets and Disks Using Projections
    on Karhunen-Loeve Eigenimages")

together with the angular-differential-imaging reference selection of

    Lafreniere et al. 2007, ApJ 660, 770 (LOCI; the ``delta_rot`` criterion)
    Marois et al. 2006, ApJ 641, 556 (ADI)

Mathematics
-----------
Let ``R`` be the ``(n_ref, n_pix)`` matrix of reference frames restricted to a
zone (annulus or annulus sector) and ``T`` the ``(n_pix,)`` target vector on the
same zone.

1. **Centring.** Every reference *and* the target have their own spatial mean
   over the zone removed (row-wise), so that the KL modes describe the
   *structure* of the speckle field, not its DC level.
2. **Gram matrix.** ``E = R R^T`` of shape ``(n_ref, n_ref)``.
   This is deliberately *not* the ``(n_pix, n_pix)`` pixel covariance: for a
   realistic zone ``n_pix >> n_ref`` and both matrices share the same non-zero
   spectrum (Soummer et al. 2012, their Eq. 5).
3. **Diagonalisation.** ``E = V diag(lambda) V^T`` via ``scipy.linalg.eigh``
   (ascending eigenvalues -- reversed here to descending).
   Eigenvalues ``<= eps * lambda_max`` are discarded: they correspond to
   numerically null directions and dividing by their square root would amplify
   round-off without bound.
4. **KL basis.** ``Z = (V / sqrt(lambda))^T R`` of shape ``(n_modes, n_pix)``.
   Each row has unit norm and the rows are mutually orthogonal, i.e.
   ``Z Z^T = I`` exactly (up to round-off).
5. **Truncated projection.** ``T_hat = (T . Z_K^T) Z_K`` with ``Z_K = Z[:K]``.
6. **Residual.** ``T - T_hat``, orthogonal to every retained KL mode.

Conventions
-----------
* Images are ``(y, x)``, cubes are ``(n_frames, y, x)``, float64 internally.
* Angles are in degrees; position angle 0 = North (``+y``), increasing towards
  East (``-x``).
* ``klip_annular`` / ``klip_fullframe`` return **non-derotated** residual cubes;
  derotation and temporal collapse are the job of :mod:`exoklip.adi`.

Notes
-----
Non-finite pixels (NaN/inf) are ignored when the zone mean is computed and are
replaced by 0 in the centred matrices, so that a bad pixel contributes nothing
to the Gram matrix instead of poisoning the whole eigendecomposition. In the
residual cubes produced by :func:`klip_annular` / :func:`klip_fullframe` those
pixels are restored to NaN.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh

from .core import frame_center, get_annulus_mask, get_segment_mask

logger = logging.getLogger(__name__)

__all__ = [
    "klip_basis",
    "klip_project",
    "klip_residual",
    "rotation_threshold_mask",
    "klip_annular",
    "klip_fullframe",
]

#: Default relative eigenvalue floor (see :func:`klip_basis`).
EPS_DEFAULT: float = 1e-12

#: Maximum number of KL bases kept in the per-zone memoisation cache.
_CACHE_SIZE: int = 8


# --------------------------------------------------------------------------- #
# private helpers                                                             #
# --------------------------------------------------------------------------- #
def _as_matrix(array: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return ``array`` as a 2D float64 array, without copying when possible.

    Parameters
    ----------
    array : array_like
        Candidate ``(n_rows, n_pix)`` matrix.
    name : str
        Argument name used in the error message.

    Returns
    -------
    numpy.ndarray
        2D float64 view or copy. Never written to by the callers.

    Raises
    ------
    ValueError
        If the array is not 2D or has zero columns.
    """
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"`{name}` must be 2D (n_rows, n_pix), got shape {arr.shape} with "
            f"{arr.ndim} dimension(s)."
        )
    if arr.shape[1] < 1:
        raise ValueError(
            f"`{name}` must have at least 1 column (pixel), got shape "
            f"{arr.shape}."
        )
    return arr


def _as_cube(cube: ArrayLike, name: str = "cube") -> NDArray[np.float64]:
    """Return ``cube`` as a 3D ``(n_frames, y, x)`` float64 array."""
    arr = np.asarray(cube, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"`{name}` must be a 3D cube (n_frames, y, x), got shape "
            f"{arr.shape} with {arr.ndim} dimension(s)."
        )
    if arr.shape[0] < 2:
        raise ValueError(
            f"`{name}` must contain at least 2 frames for a KLIP/ADI reduction, "
            f"got {arr.shape[0]} frame(s) (shape {arr.shape})."
        )
    if arr.shape[1] < 1 or arr.shape[2] < 1:
        raise ValueError(
            f"`{name}` must have strictly positive spatial dimensions, got "
            f"shape {arr.shape}."
        )
    return arr


def _as_angles(angles: ArrayLike, n_expected: int | None = None) -> NDArray[np.float64]:
    """Return the parallactic angles as a finite 1D float64 array (degrees)."""
    arr = np.asarray(angles, dtype=np.float64).ravel()
    if arr.ndim != 1 or arr.size < 1:
        raise ValueError(
            f"`angles` must be a 1D sequence of parallactic angles in degrees, "
            f"got shape {np.shape(angles)}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "`angles` must be finite; got "
            f"{int(np.count_nonzero(~np.isfinite(arr)))} non-finite value(s) "
            f"out of {arr.size}."
        )
    if n_expected is not None and arr.size != n_expected:
        raise ValueError(
            f"`angles` has {arr.size} entries but the cube has {n_expected} "
            f"frames; they must match one-to-one."
        )
    return arr


def _positive(value: Any, name: str, strict: bool = True) -> float:
    """Validate and return a finite (strictly) positive float."""
    try:
        val = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"`{name}` must be a real number, got {value!r} of type "
            f"{type(value).__name__}."
        ) from exc
    if not math.isfinite(val):
        raise ValueError(f"`{name}` must be finite, got {val}.")
    if strict and val <= 0:
        raise ValueError(f"`{name}` must be > 0, got {val}.")
    if not strict and val < 0:
        raise ValueError(f"`{name}` must be >= 0, got {val}.")
    return val


def _center_rows(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Remove the per-row (per-frame, over the zone) spatial mean.

    This is step 1 of the KLIP recipe of Soummer et al. 2012: each reference
    image *and* the target are centred over the zone before anything else.

    Parameters
    ----------
    mat : numpy.ndarray
        ``(n_rows, n_pix)`` float64 array. Never modified.

    Returns
    -------
    numpy.ndarray
        A new ``(n_rows, n_pix)`` array whose rows have zero mean over the
        finite pixels. Non-finite entries are set to 0 (they then contribute
        nothing to the Gram matrix); rows that are entirely non-finite become
        all-zero rows.
    """
    finite = np.isfinite(mat)
    if finite.all():
        return mat - mat.mean(axis=1, keepdims=True)

    logger.debug(
        "_center_rows: %d/%d non-finite pixels ignored in the zone mean and "
        "replaced by 0.", int(np.count_nonzero(~finite)), finite.size,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
        means = np.nanmean(np.where(finite, mat, np.nan), axis=1, keepdims=True)
    means = np.where(np.isfinite(means), means, 0.0)
    return np.where(finite, mat - means, 0.0)


def _n_workers(n_jobs: int) -> int:
    """Translate ``n_jobs`` (``-1`` = all CPUs) into a worker count ``>= 1``."""
    n_jobs = int(n_jobs)
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    if n_jobs < 1:
        raise ValueError(f"`n_jobs` must be >= 1 or -1 (all CPUs); got {n_jobs}.")
    return n_jobs


def _normalize_n_modes(n_modes: Any) -> tuple[list[int], bool]:
    """Normalise the ``n_modes`` argument of the reduction entry points.

    Parameters
    ----------
    n_modes : int or sequence of int
        Requested truncation rank(s).

    Returns
    -------
    k_list : list of int
        Sorted, de-duplicated list of ranks, all ``>= 1``.
    single : bool
        True if a scalar was given (the caller must return an array rather
        than a dict).
    """
    if isinstance(n_modes, bool):
        raise ValueError(f"`n_modes` must be an int or a sequence of int, got {n_modes!r}.")
    if isinstance(n_modes, (int, np.integer)):
        k = int(n_modes)
        if k < 1:
            raise ValueError(f"`n_modes` must be >= 1, got {k}.")
        return [k], True
    if isinstance(n_modes, (list, tuple, range, np.ndarray)):
        raw = np.asarray(list(n_modes)).ravel()
        if raw.size == 0:
            raise ValueError("`n_modes` sequence is empty; give at least one rank.")
        ks: list[int] = []
        for value in raw:
            k = int(value)
            if k != value or k < 1:
                raise ValueError(
                    f"every entry of `n_modes` must be an integer >= 1, got {value!r}."
                )
            ks.append(k)
        return sorted(set(ks)), False
    raise ValueError(
        f"`n_modes` must be an int or a sequence of int, got {type(n_modes).__name__}."
    )


# --------------------------------------------------------------------------- #
# public API -- linear algebra                                                #
# --------------------------------------------------------------------------- #
def klip_basis(
    references: ArrayLike,
    n_modes: int | None = None,
    eps: float = EPS_DEFAULT,
    return_eigenvalues: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Orthonormal Karhunen-Loeve basis of a set of reference frames.

    Implements Eqs. 4-6 of Soummer, Pueyo & Larkin 2012 (ApJL 755, L28):
    the references are mean-subtracted, their ``(n_ref, n_ref)`` **Gram**
    matrix ``E = R R^T`` is diagonalised, and the KL images are

    .. math::
        Z_k = \\frac{1}{\\sqrt{\\lambda_k}} \\sum_i V_{ik}\\, R_i ,

    which satisfy ``Z Z^T = I`` (unit-norm, mutually orthogonal rows).

    Parameters
    ----------
    references : array_like
        ``(n_ref, n_pix)`` matrix of reference frames restricted to a zone.
        **Not** modified; centred internally.
    n_modes : int, optional
        Number of KL modes to keep (the truncation rank ``K``). Default None =
        keep every numerically significant mode. Values larger than ``n_ref``
        are clamped to ``n_ref``.
    eps : float, optional
        Relative eigenvalue floor. Modes with ``lambda <= eps * lambda_max``
        are discarded and ``K`` is reduced accordingly (a warning is logged),
        because ``1 / sqrt(lambda)`` would otherwise amplify round-off without
        bound. Default ``1e-12``.
    return_eigenvalues : bool, optional
        If True also return the eigenvalues of the **retained** modes.

    Returns
    -------
    basis : numpy.ndarray
        ``(n_modes_kept, n_pix)`` float64 array, rows of unit L2 norm sorted by
        **decreasing** eigenvalue. ``n_modes_kept <= min(n_modes, n_ref)``.
    eigenvalues : numpy.ndarray
        Only if ``return_eigenvalues`` -- ``(n_modes_kept,)`` eigenvalues of the
        Gram matrix, in decreasing order. They are the variances carried by
        each mode (``lambda_k = ||v_k^T R||^2``).

    Raises
    ------
    ValueError
        If ``references`` is not a 2D array with at least one row and one
        column, or if ``n_modes``/``eps`` are invalid.

    Notes
    -----
    * The Gram matrix is used on purpose; the pixel covariance would be
      ``(n_pix, n_pix)`` with ``n_pix >> n_ref`` in any realistic zone.
    * ``scipy.linalg.eigh`` returns *ascending* eigenvalues; both the
      eigenvalues and the eigenvectors are reversed here.
    * Eigenvector signs are arbitrary, hence individual KL modes are defined up
      to a sign; the projector ``Z_K^T Z_K`` is not.
    * If every eigenvalue is numerically null (e.g. all references are
      constant over the zone) an empty ``(0, n_pix)`` basis is returned and a
      warning is logged; :func:`klip_project` then leaves the target untouched
      (apart from its own mean subtraction).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> refs = rng.normal(size=(8, 300))
    >>> z = klip_basis(refs, n_modes=4)
    >>> z.shape
    (4, 300)
    >>> bool(np.allclose(z @ z.T, np.eye(4), atol=1e-10))
    True
    """
    refs = _as_matrix(references, "references")
    n_ref, n_pix = refs.shape
    if n_ref < 1:
        raise ValueError(
            f"`references` must contain at least 1 reference frame, got shape "
            f"{refs.shape}."
        )
    eps = _positive(eps, "eps", strict=False)

    if n_modes is None:
        k_req = n_ref
    else:
        if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)):
            raise ValueError(
                f"`n_modes` must be an int or None, got {type(n_modes).__name__}."
            )
        k_req = int(n_modes)
        if k_req < 1:
            raise ValueError(f"`n_modes` must be >= 1 (or None), got {k_req}.")
        if k_req > n_ref:
            logger.warning(
                "klip_basis: n_modes=%d exceeds the number of references "
                "(%d); truncating to %d.", k_req, n_ref, n_ref,
            )
            k_req = n_ref

    refs_c = _center_rows(refs)
    gram = refs_c @ refs_c.T                      # (n_ref, n_ref)

    evals, evecs = eigh(gram)                     # ASCENDING eigenvalues
    evals = evals[::-1]                           # -> descending
    evecs = evecs[:, ::-1]                        # columns reordered the same way

    lam_max = float(evals[0])
    if lam_max <= 0.0:
        logger.warning(
            "klip_basis: the reference set carries no variance over this zone "
            "(largest Gram eigenvalue = %.3g <= 0); returning an empty basis.",
            lam_max,
        )
        n_valid = 0
    else:
        # evals is sorted descending, so the survivors form a prefix.
        n_valid = int(np.count_nonzero(evals > eps * lam_max))

    k = min(k_req, n_valid)
    if k < k_req:
        logger.warning(
            "klip_basis: only %d of the %d requested KL modes have an "
            "eigenvalue above the floor eps*lambda_max = %.3g; truncating to "
            "%d modes.", k, k_req, eps * lam_max, k,
        )

    lam = evals[:k]
    # Row k of `basis` is v_k^T R / sqrt(lambda_k)  ->  unit L2 norm because
    # ||v_k^T R||^2 = v_k^T (R R^T) v_k = lambda_k.
    basis = np.ascontiguousarray((evecs[:, :k] / np.sqrt(lam)).T @ refs_c)

    if return_eigenvalues:
        return basis, np.array(lam, dtype=np.float64, copy=True)
    return basis


def klip_project(
    target: ArrayLike,
    basis: ArrayLike,
    n_modes: int | None = None,
) -> NDArray[np.float64]:
    """Subtract the truncated KL projection of ``target`` onto ``basis``.

    ``residual = T_c - (T_c . Z_K^T) Z_K`` where ``T_c`` is ``target`` with its
    own spatial mean over the zone removed and ``Z_K = basis[:n_modes]``
    (Soummer, Pueyo & Larkin 2012, ApJL 755, L28, their Eq. 8).

    Parameters
    ----------
    target : array_like
        ``(n_pix,)`` single target vector or ``(m, n_pix)`` stack of targets,
        restricted to the same zone as ``basis``. **Not** modified.
    basis : array_like
        ``(n_modes_available, n_pix)`` orthonormal KL basis, typically from
        :func:`klip_basis`.
    n_modes : int, optional
        Truncation rank. Default None = use every row of ``basis``. Values
        larger than ``basis.shape[0]`` are clamped (logged at DEBUG).

    Returns
    -------
    numpy.ndarray
        Residual, ``(n_pix,)`` if ``target`` was 1D, ``(m, n_pix)`` otherwise.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or ``n_modes < 1``.

    Notes
    -----
    **The target is mean-subtracted here**, exactly like the references are
    inside :func:`klip_basis`. This is required for consistency: the KL modes
    span the space of *centred* references, so a non-centred target would leave
    an unmodellable DC component in the residual. It also makes
    :func:`klip_residual` a strict shortcut for ``klip_basis`` + ``klip_project``.

    The residual is orthogonal to every retained mode:
    ``residual @ basis[:n_modes].T == 0`` to machine precision.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> refs = rng.normal(size=(6, 200))
    >>> z = klip_basis(refs)
    >>> res = klip_project(refs[0], z)          # target is one of the refs
    >>> bool(np.linalg.norm(res) < 1e-8)
    True
    """
    z = _as_matrix(basis, "basis")
    t = np.asarray(target, dtype=np.float64)
    if t.ndim not in (1, 2):
        raise ValueError(
            f"`target` must be 1D (n_pix,) or 2D (m, n_pix), got shape "
            f"{t.shape} with {t.ndim} dimension(s)."
        )
    is_1d = t.ndim == 1
    tm = t[np.newaxis, :] if is_1d else t
    if tm.shape[1] != z.shape[1]:
        raise ValueError(
            f"`target` and `basis` must share their pixel axis: got "
            f"target n_pix={tm.shape[1]} vs basis n_pix={z.shape[1]}."
        )

    k_avail = z.shape[0]
    if n_modes is None:
        k = k_avail
    else:
        if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)):
            raise ValueError(
                f"`n_modes` must be an int or None, got {type(n_modes).__name__}."
            )
        k = int(n_modes)
        if k < 1:
            raise ValueError(f"`n_modes` must be >= 1 (or None), got {k}.")
        if k > k_avail:
            logger.debug(
                "klip_project: n_modes=%d > basis rows=%d; using %d modes.",
                k, k_avail, k_avail,
            )
            k = k_avail

    residual = _center_rows(tm)          # already a fresh array
    if k > 0:
        zk = z[:k]
        residual = residual - (residual @ zk.T) @ zk
    return residual[0] if is_1d else residual


def klip_residual(
    target: ArrayLike,
    references: ArrayLike,
    n_modes: int | None,
    eps: float = EPS_DEFAULT,
) -> NDArray[np.float64]:
    """KLIP residual of ``target`` against ``references`` -- one-shot shortcut.

    Strictly equivalent (bit-for-bit) to::

        klip_project(target, klip_basis(references, n_modes, eps))

    Parameters
    ----------
    target : array_like
        ``(n_pix,)`` or ``(m, n_pix)`` target(s) on the zone.
    references : array_like
        ``(n_ref, n_pix)`` reference matrix on the same zone.
    n_modes : int or None
        Truncation rank; None keeps every significant mode.
    eps : float, optional
        Relative eigenvalue floor, see :func:`klip_basis`.

    Returns
    -------
    numpy.ndarray
        Residual with the same shape as ``target``.

    References
    ----------
    Soummer, Pueyo & Larkin 2012, ApJL 755, L28.
    """
    basis = klip_basis(references, n_modes=n_modes, eps=eps)
    return klip_project(target, basis)


def rotation_threshold_mask(
    angles: ArrayLike,
    index: int,
    radius: float,
    fwhm: float,
    delta_rot: float = 1.0,
) -> NDArray[np.bool_]:
    """Frames usable as references for frame ``index`` at a given separation.

    A frame ``j`` is accepted when the azimuthal displacement of a putative
    companion between frames ``i = index`` and ``j`` exceeds ``delta_rot``
    resolution elements:

    .. math::
        |{\\rm PA}_j - {\\rm PA}_i| \\times r \\times \\frac{\\pi}{180}
        \\;>\\; \\delta_{\\rm rot} \\times {\\rm FWHM}

    (the left-hand side is the arc length in **pixels**). This is the LOCI /
    ADI reference-selection criterion of Lafreniere et al. 2007 (ApJ 660, 770),
    designed to bound the self-subtraction of a real companion.

    Parameters
    ----------
    angles : array_like
        ``(n_frames,)`` parallactic angles in **degrees**. They must be
        *unwrapped* (continuous, not folded into ``[-180, 180)``), otherwise a
        meridian crossing would be read as a huge rotation; see
        :func:`exoklip.io.parallactic_angles_from_headers`.
    index : int
        Index of the science frame. Negative indices are accepted
        (Python semantics).
    radius : float
        Separation at which the criterion is evaluated, in pixels, ``>= 0``
        (use the mid-radius of the annulus).
    fwhm : float
        Full width at half maximum in pixels, ``> 0``.
    delta_rot : float, optional
        Threshold in units of FWHM. Default 1.0.

    Returns
    -------
    numpy.ndarray
        ``(n_frames,)`` boolean array; True = usable as a reference.
        ``mask[index]`` is **always** False (a frame is never its own
        reference).

    Raises
    ------
    ValueError
        If ``angles`` is not a finite 1D array, ``index`` is out of range, or
        ``radius``/``fwhm``/``delta_rot`` are invalid.

    Notes
    -----
    ``delta_rot * fwhm == 0`` means *no* rotation threshold: every frame except
    ``index`` is then returned (this is what ``klip_fullframe(delta_rot=0)``
    means -- "all the other frames are references"). Without this special case
    a pair of frames sharing exactly the same PA would be rejected by the
    strict ``>``.

    If fewer references survive than the caller needs, the criterion must be
    relaxed; :func:`klip_annular` does this automatically through ``min_refs``.

    Examples
    --------
    >>> m = rotation_threshold_mask([0.0, 10.0, 20.0], 0, radius=20.0,
    ...                             fwhm=4.0, delta_rot=1.0)
    >>> m.tolist()          # 10 deg -> 3.49 px < 4 ; 20 deg -> 6.98 px > 4
    [False, False, True]
    """
    ang = _as_angles(angles)
    n = ang.size

    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError(
            f"`index` must be an integer, got {index!r} of type "
            f"{type(index).__name__}."
        )
    idx = int(index)
    if idx < 0:
        idx += n
    if not 0 <= idx < n:
        raise ValueError(
            f"`index`={index} is out of range for {n} frame(s) "
            f"(valid: {-n} .. {n - 1})."
        )

    radius = _positive(radius, "radius", strict=False)
    fwhm = _positive(fwhm, "fwhm", strict=True)
    delta_rot = _positive(delta_rot, "delta_rot", strict=False)

    threshold = delta_rot * fwhm
    if threshold == 0.0:
        mask = np.ones(n, dtype=bool)
    else:
        # Arc length in pixels swept by a companion between frame idx and j.
        displacement = np.abs(ang - ang[idx]) * radius * (math.pi / 180.0)
        mask = displacement > threshold
    mask[idx] = False
    return mask


# --------------------------------------------------------------------------- #
# private -- reference selection and zone reduction                           #
# --------------------------------------------------------------------------- #
def _select_references(
    angles: NDArray[np.float64],
    index: int,
    radius: float,
    fwhm: float,
    delta_rot: float,
    min_refs: int,
) -> tuple[NDArray[np.bool_], bool]:
    """Reference mask for one frame, with the starvation fallback.

    The ``delta_rot`` criterion of :func:`rotation_threshold_mask` is applied
    first; while fewer than ``min_refs`` frames survive it is relaxed by a
    factor 0.5, at most 3 times. As a last resort the ``min_refs`` frames with
    the largest ``|Delta PA|`` are taken.

    Parameters
    ----------
    angles : numpy.ndarray
        ``(n_frames,)`` validated parallactic angles in degrees.
    index : int
        Science frame index.
    radius, fwhm, delta_rot : float
        See :func:`rotation_threshold_mask`.
    min_refs : int
        Minimum number of references wanted (internally clipped to
        ``n_frames - 1``, the maximum available).

    Returns
    -------
    mask : numpy.ndarray
        ``(n_frames,)`` boolean reference mask, ``mask[index]`` False.
    starved : bool
        True if the last-resort fallback had to be used.
    """
    n = angles.size
    wanted = max(1, min(int(min_refs), n - 1))

    dr = float(delta_rot)
    mask = rotation_threshold_mask(angles, index, radius, fwhm, dr)
    n_relax = 0
    while int(np.count_nonzero(mask)) < wanted and n_relax < 3:
        dr *= 0.5
        n_relax += 1
        mask = rotation_threshold_mask(angles, index, radius, fwhm, dr)

    if int(np.count_nonzero(mask)) >= wanted:
        if n_relax:
            logger.debug(
                "frame %d at r=%.2f px: delta_rot relaxed %d time(s) "
                "(%.4g -> %.4g) to reach %d references.",
                index, radius, n_relax, delta_rot, dr, wanted,
            )
        return mask, False

    # Last resort: the `wanted` frames with the largest |Delta PA|.
    dpa = np.abs(angles - angles[index])
    dpa[index] = -np.inf                       # never select the frame itself
    order = np.argsort(dpa, kind="stable")[::-1][:wanted]
    mask = np.zeros(n, dtype=bool)
    mask[order] = True
    return mask, True


def _reduce_zone(
    zone: NDArray[np.float64],
    angles: NDArray[np.float64],
    radius: float,
    fwhm: float,
    delta_rot: float,
    min_refs: int,
    k_list: Sequence[int],
    eps: float,
) -> dict[int, NDArray[np.float64]]:
    """Run KLIP on one zone for every frame and every truncation rank.

    Parameters
    ----------
    zone : numpy.ndarray
        ``(n_frames, n_pix)`` cube restricted to the zone. Never modified.
    angles : numpy.ndarray
        ``(n_frames,)`` parallactic angles, degrees.
    radius : float
        Mid-radius of the zone (where the ``delta_rot`` criterion is applied).
    fwhm, delta_rot, min_refs, eps : float or int
        See :func:`klip_annular`.
    k_list : sequence of int
        Sorted truncation ranks. The eigendecomposition is computed **once**
        per distinct reference set and reused for every ``K``.

    Returns
    -------
    dict
        ``{K: (n_frames, n_pix) residuals}``.
    """
    n_frames, n_pix = zone.shape
    k_max = max(k_list)
    out = {k: np.empty((n_frames, n_pix), dtype=np.float64) for k in k_list}

    # Memoise the KL basis per distinct reference set: neighbouring frames very
    # often share one, and the result is identical by construction.
    cache: "OrderedDict[bytes, NDArray[np.float64]]" = OrderedDict()
    n_starved = 0

    for i in range(n_frames):
        mask, starved = _select_references(
            angles, i, radius, fwhm, delta_rot, min_refs
        )
        n_starved += int(starved)
        n_ref = int(np.count_nonzero(mask))
        if n_ref < 1:                                    # pragma: no cover
            raise ValueError(
                f"no reference frame available for frame {i} at r={radius:.2f} "
                f"px; the cube has only {n_frames} frames."
            )
        key = mask.tobytes()
        basis = cache.get(key)
        if basis is None:
            # Clamp K to n_ref here so that klip_basis never has to warn about
            # a benign "more modes than references" request.
            basis = klip_basis(zone[mask], n_modes=min(k_max, n_ref), eps=eps)
            cache[key] = basis
            if len(cache) > _CACHE_SIZE:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)

        target = zone[i]
        for k in k_list:
            out[k][i] = klip_project(target, basis, n_modes=k)

    bad = ~np.isfinite(zone)
    if bad.any():
        for k in k_list:
            out[k][bad] = np.nan

    if n_starved:
        wanted = max(1, min(int(min_refs), n_frames - 1))
        logger.warning(
            "KLIP zone at r=%.2f px: %d/%d frames could not gather %d "
            "reference(s) satisfying delta_rot=%.4g even after 3 relaxations "
            "(x0.5); the %d frames with the largest |dPA| were used instead. "
            "Expect self-subtraction at this separation.",
            radius, n_starved, n_frames, wanted, delta_rot, wanted,
        )
    return out


def _annulus_bounds(
    r_min: float, r_max: float, width: float
) -> NDArray[np.float64]:
    """Tile ``[r_min, r_max]`` with annuli of width as close as possible to ``width``.

    The number of annuli is ``round((r_max - r_min) / width)`` (at least 1) and
    the interval is then split into that many **equal** annuli, so that the
    whole requested radial range is processed and no ragged remainder is left
    out. The effective width therefore differs from ``width`` by at most a
    factor ~1.5.

    Returns
    -------
    numpy.ndarray
        ``(n_annuli + 1,)`` increasing radii, in pixels.
    """
    n_ann = max(1, int(math.floor((r_max - r_min) / width + 0.5)))
    return np.linspace(r_min, r_max, n_ann + 1, dtype=np.float64)


def _n_segments_for(n_segments: int | str, r_mid: float, width: float) -> int:
    """Number of azimuthal segments of one annulus."""
    if isinstance(n_segments, str):
        if n_segments.lower() != "auto":
            raise ValueError(
                f"`n_segments` must be a positive int or 'auto', got "
                f"{n_segments!r}."
            )
        return max(1, int(math.floor(2.0 * math.pi * r_mid / width)))
    if isinstance(n_segments, bool) or not isinstance(n_segments, (int, np.integer)):
        raise ValueError(
            f"`n_segments` must be a positive int or 'auto', got "
            f"{type(n_segments).__name__}."
        )
    n_seg = int(n_segments)
    if n_seg < 1:
        raise ValueError(f"`n_segments` must be >= 1 or 'auto', got {n_seg}.")
    return n_seg


# --------------------------------------------------------------------------- #
# public API -- reductions                                                    #
# --------------------------------------------------------------------------- #
def klip_annular(
    cube: ArrayLike,
    angles: ArrayLike,
    fwhm: float,
    n_modes: int | Sequence[int] = 10,
    asize: float = 4.0,
    delta_rot: float = 1.0,
    n_segments: int | str = 1,
    r_min: float | None = None,
    r_max: float | None = None,
    min_refs: int = 5,
    center: Sequence[float] | None = None,
    verbose: bool = False,
    n_jobs: int = 1,
) -> NDArray[np.float64] | dict[int, NDArray[np.float64]]:
    """Annular KLIP-ADI -- the reference reduction mode.

    The frame is tiled into concentric annuli of width ``asize * fwhm`` pixels,
    optionally split into azimuthal segments. For every frame and every zone,
    the reference library is the set of frames passing the ``delta_rot``
    criterion of :func:`rotation_threshold_mask` (evaluated at the mid-radius of
    the annulus), a KL basis is built from them
    (Soummer, Pueyo & Larkin 2012, ApJL 755, L28) and the truncated projection
    is subtracted from the frame.

    Parameters
    ----------
    cube : array_like
        ``(n_frames, y, x)`` science cube, at least 2 frames. Never modified.
    angles : array_like
        ``(n_frames,)`` parallactic angles in degrees, unwrapped.
    fwhm : float
        FWHM in pixels, ``> 0``.
    n_modes : int or sequence of int, optional
        Truncation rank ``K``. If a sequence is given, the **same**
        eigendecomposition is reused for every ``K`` (which is the whole point
        of a truncated KL expansion) and a dict is returned. Default 10.
    asize : float, optional
        Annulus width in units of FWHM. Default 4.0.
    delta_rot : float, optional
        Rotation threshold in units of FWHM. Default 1.0.
    n_segments : int or {'auto'}, optional
        Number of azimuthal segments per annulus. ``'auto'`` gives
        ``max(1, floor(2 * pi * r_mid / (asize * fwhm)))``, i.e. segments about
        as long as the annulus is wide. Default 1 (full annuli).
    r_min, r_max : float, optional
        Radial range in pixels. Defaults: ``r_min = fwhm`` and ``r_max`` = the
        largest radius fully inscribed in the frame.
    min_refs : int, optional
        Minimum size of the reference library. If the ``delta_rot`` criterion
        yields fewer, it is relaxed by 0.5 at most 3 times, then the
        ``min_refs`` frames with the largest ``|Delta PA|`` are taken and a
        warning is logged. Default 5.
    center : sequence of float, optional
        ``(cy, cx)``. Defaults to :func:`exoklip.core.frame_center`.
    verbose : bool, optional
        Log the zone-by-zone progress at INFO level (configure ``logging`` to
        see it; this package never prints). Default False.
    n_jobs : int, optional
        Number of threads used over the annuli (``-1`` = all CPUs). NumPy/BLAS
        releases the GIL, so threads do help here. Default 1.

    Returns
    -------
    numpy.ndarray or dict
        ``(n_frames, y, x)`` cube of **non-derotated** residuals if ``n_modes``
        is an int, else ``{K: cube}``. Pixels that belong to no processed zone
        (and pixels that were non-finite in the input) are NaN. Derotation and
        collapse are done by :mod:`exoklip.adi`.

    Raises
    ------
    ValueError
        On any inconsistent shape or invalid parameter (the message states what
        was received and what was expected).

    Notes
    -----
    The zone geometry is fixed in the **detector** frame, which is exactly what
    makes ADI work: the quasi-static speckles stay put while a companion moves
    azimuthally by ``Delta PA * r`` pixels.

    Because each residual has the zone mean of its own frame removed, the
    residual cube has (approximately) zero mean over every zone.

    References
    ----------
    Soummer, Pueyo & Larkin 2012, ApJL 755, L28.
    Lafreniere et al. 2007, ApJ 660, 770 (the ``delta_rot`` criterion).
    Marois et al. 2006, ApJ 641, 556 (ADI).
    """
    arr = _as_cube(cube)
    n_frames, ny, nx = arr.shape
    ang = _as_angles(angles, n_expected=n_frames)
    fwhm = _positive(fwhm, "fwhm")
    asize = _positive(asize, "asize")
    delta_rot = _positive(delta_rot, "delta_rot", strict=False)
    k_list, single = _normalize_n_modes(n_modes)

    if isinstance(min_refs, bool) or not isinstance(min_refs, (int, np.integer)):
        raise ValueError(
            f"`min_refs` must be an int >= 1, got {type(min_refs).__name__}."
        )
    min_refs = int(min_refs)
    if min_refs < 1:
        raise ValueError(f"`min_refs` must be >= 1, got {min_refs}.")

    if center is None:
        cy, cx = frame_center((ny, nx))
    else:
        c = np.asarray(center, dtype=float).ravel()
        if c.size != 2 or not np.all(np.isfinite(c)):
            raise ValueError(
                f"`center` must be a finite 2-element sequence (cy, cx), got "
                f"{center!r}."
            )
        cy, cx = float(c[0]), float(c[1])

    # Largest radius fully inscribed in the frame.
    r_edge = float(min(cy, cx, (ny - 1) - cy, (nx - 1) - cx))
    rin = fwhm if r_min is None else _positive(r_min, "r_min", strict=False)
    rout = r_edge if r_max is None else _positive(r_max, "r_max")
    if rout <= rin:
        raise ValueError(
            f"`r_max` must be > `r_min`: got r_min={rin:.3f}, r_max={rout:.3f} "
            f"px for a {ny}x{nx} frame centred on ({cy:.2f}, {cx:.2f}) "
            f"(inscribed radius {r_edge:.2f} px). Give explicit r_min/r_max or "
            f"use a larger frame."
        )

    width = asize * fwhm
    bounds = _annulus_bounds(rin, rout, width)
    n_ann = bounds.size - 1
    if verbose:
        logger.info(
            "klip_annular: %d frame(s) of %dx%d, %d annuli from %.2f to %.2f "
            "px (target width %.2f px, effective %.2f px), K=%s, "
            "delta_rot=%.3g, n_segments=%r.",
            n_frames, ny, nx, n_ann, rin, rout, width,
            float(bounds[1] - bounds[0]), k_list, delta_rot, n_segments,
        )

    shape2d = (ny, nx)
    ctr = (cy, cx)

    def _one_annulus(a: int) -> list[tuple[NDArray[np.intp], NDArray[np.intp],
                                           dict[int, NDArray[np.float64]]]]:
        """Reduce every segment of annulus ``a``; returns (ys, xs, residuals)."""
        r_in = float(bounds[a])
        r_out = float(bounds[a + 1])
        r_mid = 0.5 * (r_in + r_out)
        n_seg = _n_segments_for(n_segments, r_mid, width)
        results = []
        for s in range(n_seg):
            if n_seg == 1:
                mask2d = get_annulus_mask(shape2d, r_in, r_out, center=ctr)
            else:
                pa0 = s * 360.0 / n_seg
                pa1 = (s + 1) * 360.0 / n_seg
                mask2d = get_segment_mask(
                    shape2d, r_in, r_out, pa0, pa1, center=ctr
                )
            ys, xs = np.nonzero(mask2d)
            if ys.size == 0:
                logger.debug(
                    "empty zone: annulus %d [%.2f, %.2f) px, segment %d/%d.",
                    a, r_in, r_out, s + 1, n_seg,
                )
                continue
            zone = arr[:, ys, xs]                  # (n_frames, n_pix) copy
            res = _reduce_zone(
                zone, ang, r_mid, fwhm, delta_rot, min_refs, k_list, EPS_DEFAULT
            )
            results.append((ys, xs, res))
        if verbose:
            logger.info(
                "  annulus %d/%d: r=[%.2f, %.2f) px, %d segment(s), "
                "%d pixel(s).",
                a + 1, n_ann, r_in, r_out, n_seg,
                int(sum(r[0].size for r in results)),
            )
        return results

    workers = _n_workers(n_jobs)
    if workers > 1 and n_ann > 1:
        with ThreadPoolExecutor(max_workers=min(workers, n_ann)) as pool:
            all_results = list(pool.map(_one_annulus, range(n_ann)))
    else:
        all_results = [_one_annulus(a) for a in range(n_ann)]

    out = {
        k: np.full((n_frames, ny, nx), np.nan, dtype=np.float64) for k in k_list
    }
    n_pix_done = 0
    for annulus_results in all_results:
        for ys, xs, res in annulus_results:
            n_pix_done += ys.size
            for k in k_list:
                out[k][:, ys, xs] = res[k]

    if n_pix_done == 0:
        raise ValueError(
            f"no pixel was processed: the radial range [{rin:.2f}, {rout:.2f}] "
            f"px does not intersect the {ny}x{nx} frame centred on "
            f"({cy:.2f}, {cx:.2f})."
        )
    if verbose:
        logger.info(
            "klip_annular: done, %d/%d pixel(s) per frame processed "
            "(%.1f%%).", n_pix_done, ny * nx, 100.0 * n_pix_done / (ny * nx),
        )

    return out[k_list[0]] if single else out


def klip_fullframe(
    cube: ArrayLike,
    angles: ArrayLike,
    fwhm: float,
    n_modes: int | Sequence[int] = 10,
    delta_rot: float = 0.0,
    mask_radius: float | None = None,
    center: Sequence[float] | None = None,
) -> NDArray[np.float64] | dict[int, NDArray[np.float64]]:
    """Full-frame PCA/KLIP -- one single zone covering the whole image.

    Faster than :func:`klip_annular` (a single eigendecomposition per distinct
    reference set instead of one per annulus) but less performant at small
    separation, where a single set of KL modes cannot describe both the bright
    inner speckles and the faint outer halo.

    With the default ``delta_rot=0`` there is no rotation threshold at all:
    every *other* frame is a reference, which maximises speckle suppression and
    self-subtraction alike.

    Parameters
    ----------
    cube : array_like
        ``(n_frames, y, x)`` science cube. Never modified.
    angles : array_like
        ``(n_frames,)`` parallactic angles in degrees, unwrapped.
    fwhm : float
        FWHM in pixels, ``> 0``.
    n_modes : int or sequence of int, optional
        Truncation rank(s); a sequence returns ``{K: cube}`` reusing the same
        eigendecomposition. Default 10.
    delta_rot : float, optional
        Rotation threshold in units of FWHM, evaluated at the inner working
        radius (``mask_radius`` if given and larger than ``fwhm``, else
        ``fwhm``) -- the most conservative choice for a zone that spans every
        separation. Default 0.0 (no threshold).
    mask_radius : float, optional
        Blank the central disc of this radius (pixels): those pixels are
        excluded from the KL fit and set to NaN in the output. Default None.
    center : sequence of float, optional
        ``(cy, cx)``. Defaults to :func:`exoklip.core.frame_center`.

    Returns
    -------
    numpy.ndarray or dict
        ``(n_frames, y, x)`` **non-derotated** residual cube, or ``{K: cube}``.

    Raises
    ------
    ValueError
        On inconsistent shapes or invalid parameters.

    Notes
    -----
    The reference-starvation fallback of :func:`klip_annular` is applied here
    too, with an internal ``min_refs = min(5, n_frames - 1)``; it only ever
    triggers for ``delta_rot > 0``.

    References
    ----------
    Soummer, Pueyo & Larkin 2012, ApJL 755, L28.
    """
    arr = _as_cube(cube)
    n_frames, ny, nx = arr.shape
    ang = _as_angles(angles, n_expected=n_frames)
    fwhm = _positive(fwhm, "fwhm")
    delta_rot = _positive(delta_rot, "delta_rot", strict=False)
    k_list, single = _normalize_n_modes(n_modes)

    if center is None:
        cy, cx = frame_center((ny, nx))
    else:
        c = np.asarray(center, dtype=float).ravel()
        if c.size != 2 or not np.all(np.isfinite(c)):
            raise ValueError(
                f"`center` must be a finite 2-element sequence (cy, cx), got "
                f"{center!r}."
            )
        cy, cx = float(c[0]), float(c[1])

    if mask_radius is None:
        mask2d = np.ones((ny, nx), dtype=bool)
        r_eval = fwhm
    else:
        m_rad = _positive(mask_radius, "mask_radius", strict=False)
        r_corner = float(np.hypot(max(cy, ny - 1 - cy), max(cx, nx - 1 - cx)))
        if m_rad >= r_corner:
            raise ValueError(
                f"`mask_radius`={m_rad:.3f} px masks the whole {ny}x{nx} frame "
                f"(largest radius {r_corner:.3f} px); nothing left to reduce."
            )
        # r >= mask_radius, i.e. the complement of the central disc.
        mask2d = get_annulus_mask((ny, nx), m_rad, r_corner + 1.0, center=(cy, cx))
        r_eval = max(m_rad, fwhm)

    ys, xs = np.nonzero(mask2d)
    if ys.size < 1:                                        # pragma: no cover
        raise ValueError(
            f"the full-frame zone is empty for a {ny}x{nx} frame with "
            f"mask_radius={mask_radius!r}."
        )

    zone = arr[:, ys, xs]
    res = _reduce_zone(
        zone, ang, r_eval, fwhm, delta_rot,
        min_refs=max(1, min(5, n_frames - 1)),
        k_list=k_list, eps=EPS_DEFAULT,
    )

    out = {
        k: np.full((n_frames, ny, nx), np.nan, dtype=np.float64) for k in k_list
    }
    for k in k_list:
        out[k][:, ys, xs] = res[k]
    return out[k_list[0]] if single else out
