"""FITS and legacy image I/O, and parallactic-angle bookkeeping.

This module is the boundary between the outside world and the rest of
:mod:`exoklip`.  It has **no hard dependency beyond numpy**: FITS support is
provided by ``astropy`` imported *lazily* inside each function, and JPEG
support by ``Pillow`` imported the same way.  PNG is decoded natively with
:mod:`zlib` and :mod:`struct` so that the legacy path of the package works in a
bare numpy/scipy environment.

Conventions
-----------
* Images are ``(y, x)``, cubes are ``(n_frames, y, x)``, ``float64`` internally.
* Angles are in **degrees** in every public API.
* Position angles returned by :func:`parallactic_angles_from_headers` are
  *unwrapped*: the 180/-180 discontinuity is removed so that consecutive
  differences (used by the rotation threshold of :mod:`exoklip.klip`) are the
  true field-rotation increments.  Values may therefore fall outside
  ``[-180, 180]`` by design.

References
----------
Meeus 1998, *Astronomical Algorithms*, 2nd ed., ch. 14 -- parallactic angle.
Smart 1962, *Text-book on Spherical Astronomy*, ch. 2 -- the PZX triangle.
Yelda et al. 2010, ApJ 725, 331 -- Keck/NIRC2 astrometric calibration and the
``PARANG + ROTPOSN - INSTANGL`` position-angle combination.
W3C/ISO 15948:2004 -- PNG (Portable Network Graphics) specification, used for
the native decoder (chunk layout, filter types, sample packing).
"""

from __future__ import annotations

import glob as _glob
import logging
import math
import os
import struct
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "load_fits",
    "save_fits",
    "load_cube_from_dir",
    "parallactic_angle",
    "parallactic_angles_from_headers",
    "load_image_legacy",
    "NIRC2_PRESET",
    "INSTRUMENT_PRESETS",
]

logger = logging.getLogger(__name__)

_ASTROPY_MSG = "astropy is required for FITS I/O: pip install astropy"

#: Header keywords tried, in order, when ``keys is None`` and no instrument
#: preset matched.  Each one is expected to hold the position angle directly.
_DEFAULT_PA_KEYS: tuple[str, ...] = ("PARANG", "PA")

#: Keywords searched for the site latitude when reconstructing the parallactic
#: angle from the hour angle and declination.
_LATITUDE_KEYS: tuple[str, ...] = (
    "LATITUDE",
    "SITELAT",
    "OBSLAT",
    "TELLAT",
    "HIERARCH ESO TEL GEOLAT",
    "ESO TEL GEOLAT",
)

#: Keywords searched for the hour angle (hours) and the declination (degrees).
_HA_KEYS: tuple[str, ...] = ("HA", "HOURANG", "HRANGLE")
_DEC_KEYS: tuple[str, ...] = ("DEC", "DEC_D", "OBJDEC", "TELDEC", "CRVAL2")
_LST_KEYS: tuple[str, ...] = ("LST", "ST", "SIDEREAL")
_RA_KEYS: tuple[str, ...] = ("RA", "RA_D", "OBJRA", "TELRA", "CRVAL1")

#: Keck II / NIRC2 preset.
#:
#: In *vertical angle* (pupil-tracking) mode the on-sky position angle of the
#: NIRC2 detector is ``PARANG + ROTPOSN - INSTANGL``, with ``INSTANGL = 0.7``
#: deg the fixed instrument angle of the camera (Yelda et al. 2010, ApJ 725,
#: 331).  The latitude is that of the Keck observatory on Maunakea.
NIRC2_PRESET: dict[str, Any] = {
    "instrument": "NIRC2",
    "telescope": "Keck II",
    "latitude": 19.8283,      # deg North
    "longitude": -155.4783,   # deg East (negative = West)
    "add": ("PARANG", "ROTPOSN"),
    "sub": ("INSTANGL",),
    "instangl_default": 0.7,  # deg, used only if INSTANGL is absent
    "ha_key": "HA",
    "dec_key": "DEC",
    "instrume_match": ("NIRC2",),
    "reference": "Yelda et al. 2010, ApJ 725, 331",
}

#: Registry of instrument presets, keyed by lowercase alias.
INSTRUMENT_PRESETS: dict[str, dict[str, Any]] = {
    "nirc2": NIRC2_PRESET,
    "keck": NIRC2_PRESET,
    "keck2": NIRC2_PRESET,
    "keck/nirc2": NIRC2_PRESET,
}


# --------------------------------------------------------------------------- #
# private helpers -- optional imports                                          #
# --------------------------------------------------------------------------- #
def _import_fits() -> Any:
    """Import ``astropy.io.fits`` lazily.

    Returns
    -------
    module
        The ``astropy.io.fits`` module.

    Raises
    ------
    ImportError
        If astropy is not installed, with an actionable install hint.
    """
    try:
        from astropy.io import fits  # noqa: PLC0415  (lazy on purpose)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_ASTROPY_MSG) from exc
    return fits


def _as_path(path: str | os.PathLike[str], argname: str = "path") -> str:
    """Validate and normalise a filesystem path."""
    if isinstance(path, (str, os.PathLike)):
        return os.fspath(path)
    raise ValueError(
        f"`{argname}` must be a str or os.PathLike, got {type(path).__name__}."
    )


# --------------------------------------------------------------------------- #
# private helpers -- header access                                             #
# --------------------------------------------------------------------------- #
_MISSING = object()


def _header_get(header: Any, key: str, default: Any = None) -> Any:
    """Case-insensitive lookup in a FITS header or a plain mapping.

    ``astropy.io.fits.Header`` is already case-insensitive; plain :class:`dict`
    is not, so the key is also tried upper-cased, lower-cased and finally
    through a case-insensitive scan.

    Parameters
    ----------
    header : mapping
        A ``fits.Header`` or any mapping from keyword to value.
    key : str
        Keyword to look up.
    default : object, optional
        Value returned when the keyword is absent (or holds ``None`` / an
        ``astropy`` undefined card).

    Returns
    -------
    object
        The header value, or ``default``.
    """
    if header is None:
        return default
    for candidate in (key, key.upper(), key.lower()):
        try:
            value = header[candidate]
        except (KeyError, TypeError, IndexError):
            continue
        if value is not None:
            return value
    # Last resort: linear case-insensitive scan (plain dicts with odd casing).
    try:
        items = header.items()
    except AttributeError:
        return default
    target = key.upper().strip()
    for name, value in items:
        if isinstance(name, str) and name.upper().strip() == target:
            if value is not None:
                return value
    return default


def _sexagesimal_to_float(text: str) -> float:
    """Parse ``'dd:mm:ss.s'``/``'dd mm ss.s'`` (or a plain number) to a float.

    The sign carried by the first field applies to the whole value, so
    ``'-00:30:00'`` correctly yields ``-0.5``.

    Parameters
    ----------
    text : str
        Sexagesimal or decimal string.

    Returns
    -------
    float
        Value in the unit of the first field (hours or degrees).

    Raises
    ------
    ValueError
        If the string cannot be parsed.
    """
    cleaned = text.strip().replace(":", " ").replace("d", " ").replace("h", " ")
    cleaned = cleaned.replace("m", " ").replace("s", " ").strip()
    if not cleaned:
        raise ValueError(f"cannot parse an empty angle string: {text!r}")
    parts = cleaned.split()
    if len(parts) > 3:
        raise ValueError(f"cannot parse {text!r} as a sexagesimal value.")
    sign = -1.0 if parts[0].lstrip().startswith("-") else 1.0
    total = 0.0
    for i, part in enumerate(parts):
        total += abs(float(part)) / (60.0 ** i)
    return sign * total


def _to_float(value: Any, what: str) -> float:
    """Coerce a header value (number, numeric string, sexagesimal) to a float.

    Parameters
    ----------
    value : object
        Raw header value.
    what : str
        Name used in the error message.

    Returns
    -------
    float
        The parsed value.

    Raises
    ------
    ValueError
        If the value cannot be interpreted as a finite number.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{what} has a boolean value ({value!r}), expected a number.")
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
    elif isinstance(value, str):
        try:
            out = float(value.strip())
        except ValueError:
            out = _sexagesimal_to_float(value)
    else:
        raise ValueError(
            f"{what} has type {type(value).__name__}, expected a number or a "
            f"numeric/sexagesimal string."
        )
    if not math.isfinite(out):
        raise ValueError(f"{what} is not finite (got {out}).")
    return out


def _get_float(header: Any, keys: Sequence[str], what: str) -> float | None:
    """First parsable float among ``keys`` in ``header``, or None."""
    for key in keys:
        raw = _header_get(header, key, _MISSING)
        if raw is _MISSING or raw is None:
            continue
        try:
            return _to_float(raw, f"{what} (keyword {key!r})")
        except ValueError as exc:
            logger.debug("ignoring keyword %r: %s", key, exc)
    return None


def _check_headers(headers: Any) -> list[Any]:
    """Validate the ``headers`` argument and return it as a list."""
    if headers is None:
        raise ValueError("`headers` must be a sequence of headers, got None.")
    if isinstance(headers, Mapping):
        raise ValueError(
            "`headers` must be a *sequence* of headers (one per frame), got a "
            "single mapping. Wrap it in a list: [header]."
        )
    try:
        out = list(headers)
    except TypeError as exc:
        raise ValueError(
            f"`headers` must be an iterable of headers, got "
            f"{type(headers).__name__}."
        ) from exc
    if len(out) == 0:
        raise ValueError("`headers` is empty; at least one header is required.")
    return out


# --------------------------------------------------------------------------- #
# parallactic angles                                                           #
# --------------------------------------------------------------------------- #
def parallactic_angle(
    ha_hours: ArrayLike,
    dec_deg: ArrayLike,
    lat_deg: ArrayLike,
) -> np.float64 | NDArray[np.float64]:
    r"""Parallactic angle of a target, in degrees.

    .. math::
        q = \arctan2\!\left(\sin H,\;
            \cos\delta\,\tan\varphi - \sin\delta\,\cos H\right)

    with :math:`H = 15^\circ \times \mathrm{ha\_hours}` the hour angle,
    :math:`\delta` the declination and :math:`\varphi` the geodetic latitude of
    the site (Meeus 1998, *Astronomical Algorithms*, 2nd ed., eq. 14.1; derived
    from the PZX spherical triangle, Smart 1962, ch. 2).

    ``q`` is the angle at the target between the direction to the North
    celestial pole and the direction to the zenith, counted from North towards
    East -- i.e. exactly the rate at which the sky field rotates on a
    pupil-stabilised (ADI) detector.  It is 0 at transit for a target south of
    the zenith, and 180 deg at transit for a target north of it.

    Parameters
    ----------
    ha_hours : array_like
        Hour angle in **hours** (not degrees); negative before transit.
        Broadcast against the other arguments.
    dec_deg : array_like
        Declination in degrees, within ``[-90, 90]``.
    lat_deg : array_like
        Geodetic latitude of the observatory in degrees, within ``(-90, 90)``.
        (Keck: ``19.8283``; see :data:`NIRC2_PRESET`.)

    Returns
    -------
    numpy.float64 or numpy.ndarray
        Parallactic angle(s) in degrees, in ``(-180, 180]``.  A scalar
        ``numpy.float64`` (which *is* a Python ``float``) is returned when all
        inputs are scalars, otherwise a float64 array of the broadcast shape.

    Raises
    ------
    ValueError
        If any input is non-finite, if ``|dec| > 90``, or if ``|lat| >= 90``
        (``tan(lat)`` diverges at the geographic poles).

    Notes
    -----
    At the zenith (``dec == lat`` and ``H == 0``) both arguments of
    ``arctan2`` vanish and the parallactic angle is genuinely undefined;
    ``numpy.arctan2(0, 0)`` returns 0 and that value is passed through.

    Examples
    --------
    >>> float(parallactic_angle(0.0, 0.0, 45.0))          # transit, south
    0.0
    >>> float(parallactic_angle(6.0, 0.0, 45.0))          # H = 90 deg
    45.0
    >>> float(parallactic_angle(0.0, 60.0, 45.0))         # transit, north
    180.0
    """
    ha = np.asarray(ha_hours, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64)

    for arr, name in ((ha, "ha_hours"), (dec, "dec_deg"), (lat, "lat_deg")):
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"`{name}` must be finite, got {arr.size} value(s) of which "
                f"{int(np.sum(~np.isfinite(arr)))} are not finite."
            )
    if np.any(np.abs(dec) > 90.0):
        raise ValueError(
            f"`dec_deg` must lie in [-90, 90] degrees, got a maximum "
            f"|dec| of {float(np.max(np.abs(dec)))}."
        )
    if np.any(np.abs(lat) >= 90.0):
        raise ValueError(
            f"`lat_deg` must lie strictly inside (-90, 90) degrees (tan(lat) "
            f"diverges at the poles), got a maximum |lat| of "
            f"{float(np.max(np.abs(lat)))}."
        )
    try:
        np.broadcast_shapes(ha.shape, dec.shape, lat.shape)
    except ValueError as exc:
        raise ValueError(
            f"`ha_hours` {ha.shape}, `dec_deg` {dec.shape} and `lat_deg` "
            f"{lat.shape} are not broadcast-compatible."
        ) from exc

    h_rad = np.radians(ha * 15.0)  # 1 hour of hour angle = 15 degrees
    dec_rad = np.radians(dec)
    lat_rad = np.radians(lat)

    numerator = np.sin(h_rad)
    denominator = np.cos(dec_rad) * np.tan(lat_rad) - np.sin(dec_rad) * np.cos(h_rad)
    out = np.degrees(np.arctan2(numerator, denominator))

    if out.ndim == 0:
        return np.float64(out[()])
    return np.asarray(out, dtype=np.float64)


def _unwrap_degrees(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Remove the 180/-180 wrap of a position-angle sequence.

    The unwrapping is done on **radians** with :func:`numpy.unwrap` (default
    discontinuity of pi) and converted back to degrees, so that consecutive
    differences never jump by 360 deg.  A sequence of length 1 is returned
    unchanged.
    """
    if angles.size < 2:
        return angles
    return np.degrees(np.unwrap(np.radians(angles)))


def _combination_angles(
    headers: Sequence[Any],
    add: Sequence[str],
    sub: Sequence[str],
    defaults: Mapping[str, float] | None = None,
) -> NDArray[np.float64] | None:
    """Evaluate ``sum(add) - sum(sub)`` over every header, or None.

    Returns None as soon as one *required* keyword (the first of ``add``) is
    missing from any header.  Missing optional keywords fall back to
    ``defaults`` (with a warning) or to 0.
    """
    if not add:
        return None
    out = np.zeros(len(headers), dtype=np.float64)
    defaults = dict(defaults or {})
    warned: set[str] = set()
    for i, header in enumerate(headers):
        total = 0.0
        for j, key in enumerate(add):
            value = _get_float(header, (key,), f"header[{i}][{key!r}]")
            if value is None:
                if j == 0:
                    return None
                value = float(defaults.get(key.upper(), 0.0))
                if key not in warned:
                    warned.add(key)
                    logger.warning(
                        "keyword %r missing from the headers, assuming %.4f deg.",
                        key, value,
                    )
            total += value
        for key in sub:
            value = _get_float(header, (key,), f"header[{i}][{key!r}]")
            if value is None:
                value = float(defaults.get(key.upper(), 0.0))
                if key not in warned:
                    warned.add(key)
                    logger.warning(
                        "keyword %r missing from the headers, assuming %.4f deg.",
                        key, value,
                    )
            total -= value
        out[i] = total
    return out


def _single_key_angles(
    headers: Sequence[Any], key: str
) -> NDArray[np.float64] | None:
    """Read ``key`` from every header, or None if it is missing from any."""
    out = np.empty(len(headers), dtype=np.float64)
    for i, header in enumerate(headers):
        value = _get_float(header, (key,), f"header[{i}][{key!r}]")
        if value is None:
            return None
        out[i] = value
    return out


def _reconstructed_angles(
    headers: Sequence[Any],
    latitude: float | None,
    ha_key: str | None = None,
    dec_key: str | None = None,
) -> NDArray[np.float64] | None:
    """Rebuild the parallactic angle from HA/DEC/latitude, or None."""
    ha_keys = (ha_key,) + _HA_KEYS if ha_key else _HA_KEYS
    dec_keys = (dec_key,) + _DEC_KEYS if dec_key else _DEC_KEYS
    out = np.empty(len(headers), dtype=np.float64)
    for i, header in enumerate(headers):
        ha = _get_float(header, ha_keys, f"header[{i}] hour angle")
        if ha is None:
            lst = _get_float(header, _LST_KEYS, f"header[{i}] LST")
            ra = _get_float(header, _RA_KEYS, f"header[{i}] RA")
            if lst is None or ra is None:
                return None
            # RA is stored in degrees by most instruments; convert to hours.
            ha = lst - (ra / 15.0 if abs(ra) > 24.0 else ra)
        dec = _get_float(header, dec_keys, f"header[{i}] declination")
        if dec is None:
            return None
        lat = latitude
        if lat is None:
            lat = _get_float(header, _LATITUDE_KEYS, f"header[{i}] latitude")
        if lat is None:
            return None
        out[i] = float(parallactic_angle(ha, dec, lat))
    return out


def parallactic_angles_from_headers(
    headers: Sequence[Any],
    keys: str | Sequence[str] | Mapping[str, Any] | None = None,
    latitude: float | None = None,
) -> NDArray[np.float64]:
    """Extract (or reconstruct) the parallactic angle of every frame.

    The returned angles are **unwrapped**: the 180/-180 discontinuity crossed
    by targets transiting near the zenith is removed with :func:`numpy.unwrap`
    applied to the angles in radians.  A 360 deg offset is irrelevant for a
    derotation but *not* for the ``|PA_j - PA_i|`` differences used by
    :func:`exoklip.klip.rotation_threshold_mask`, which would otherwise see a
    spurious 360 deg field rotation in the middle of the sequence.

    Search strategy when ``keys is None``
    -------------------------------------
    1. If ``INSTRUME``/``CURRINST`` identifies a known instrument, its preset
       is used (currently only Keck/NIRC2, see :data:`NIRC2_PRESET`:
       ``PARANG + ROTPOSN - INSTANGL``).
    2. Otherwise ``PARANG``, then ``PA``, read directly.
    3. Otherwise the Keck/NIRC2 combination ``PARANG + ROTPOSN - INSTANGL``.
    4. Otherwise the angle is recomputed with :func:`parallactic_angle` from
       the hour angle (``HA``, or ``LST - RA``), the declination (``DEC``) and
       the latitude (argument ``latitude``, else a header keyword such as
       ``LATITUDE``/``SITELAT``/``ESO TEL GEOLAT``).

    A strategy is only accepted if it succeeds for **every** header.

    Parameters
    ----------
    headers : sequence of mapping
        One header per frame: ``astropy.io.fits.Header`` objects or plain
        dictionaries (keyword lookup is case-insensitive in both cases).
    keys : str or sequence of str or mapping, optional
        Overrides the search strategy:

        * ``str`` -- either an instrument preset alias (``'nirc2'``,
          ``'keck'``) or a single header keyword to read;
        * sequence of ``str`` -- keywords tried in order, each read directly;
        * mapping -- an explicit linear combination
          ``{'add': ('PARANG', 'ROTPOSN'), 'sub': ('INSTANGL',)}``; the
          optional entries ``'latitude'``, ``'ha_key'``, ``'dec_key'`` and
          ``'instangl_default'`` are honoured as in :data:`NIRC2_PRESET`.
    latitude : float, optional
        Site latitude in degrees, used only by the HA/DEC reconstruction.
        Takes precedence over the preset and over any header keyword.

    Returns
    -------
    numpy.ndarray
        ``(n_frames,)`` float64 array of unwrapped position angles in degrees,
        in the same order as ``headers``.

    Raises
    ------
    ValueError
        If ``headers`` is empty/not a sequence, if ``keys`` has an unsupported
        type, or if no strategy could produce an angle for every frame (the
        message lists what was searched).

    Examples
    --------
    >>> h = [{'PARANG': 179.0}, {'PARANG': -179.0}]
    >>> parallactic_angles_from_headers(h)
    array([179., 181.])
    """
    hdrs = _check_headers(headers)
    if latitude is not None:
        latitude = _to_float(latitude, "`latitude`")
        if abs(latitude) >= 90.0:
            raise ValueError(
                f"`latitude` must lie strictly inside (-90, 90) degrees, got "
                f"{latitude}."
            )

    tried: list[str] = []
    angles: NDArray[np.float64] | None = None
    strategy = ""

    # ---- explicit user request ------------------------------------------- #
    if isinstance(keys, Mapping):
        preset = dict(keys)
        add = tuple(preset.get("add", ()) or ())
        sub = tuple(preset.get("sub", ()) or ())
        if not add:
            raise ValueError(
                "`keys` given as a mapping must contain a non-empty 'add' "
                f"entry, got {sorted(preset)}."
            )
        defaults = {"INSTANGL": float(preset.get("instangl_default", 0.0))}
        angles = _combination_angles(hdrs, add, sub, defaults)
        strategy = f"combination {'+'.join(add)}" + (
            f"-{'-'.join(sub)}" if sub else ""
        )
        tried.append(strategy)
        if angles is None:
            lat = latitude if latitude is not None else preset.get("latitude")
            angles = _reconstructed_angles(
                hdrs, lat, preset.get("ha_key"), preset.get("dec_key")
            )
            strategy = "HA/DEC/latitude reconstruction"
            tried.append(strategy)
    elif isinstance(keys, str):
        alias = keys.strip().lower()
        if alias in INSTRUMENT_PRESETS:
            return parallactic_angles_from_headers(
                hdrs, keys=INSTRUMENT_PRESETS[alias], latitude=latitude
            )
        angles = _single_key_angles(hdrs, keys)
        strategy = f"keyword {keys!r}"
        tried.append(strategy)
    elif isinstance(keys, Sequence):
        for key in keys:
            if not isinstance(key, str):
                raise ValueError(
                    f"`keys` must contain header keyword strings, got an "
                    f"element of type {type(key).__name__}."
                )
            angles = _single_key_angles(hdrs, key)
            tried.append(f"keyword {key!r}")
            if angles is not None:
                strategy = f"keyword {key!r}"
                break
    elif keys is not None:
        raise ValueError(
            f"`keys` must be None, a str (preset alias or keyword), a sequence "
            f"of str, or a mapping describing a combination; got "
            f"{type(keys).__name__}."
        )
    else:
        # ---- automatic strategy ------------------------------------------ #
        instrument = ""
        for key in ("INSTRUME", "CURRINST", "INSTRUMENT"):
            raw = _header_get(hdrs[0], key)
            if isinstance(raw, str) and raw.strip():
                instrument = raw.strip().upper()
                break
        preset: dict[str, Any] | None = None
        for alias, candidate in INSTRUMENT_PRESETS.items():
            del alias
            matches = candidate.get("instrume_match", ())
            if instrument and any(m.upper() in instrument for m in matches):
                preset = candidate
                break
        if preset is not None:
            logger.info(
                "headers identify instrument %r: using the %s preset (%s).",
                instrument, preset["instrument"], preset["reference"],
            )
            angles = _combination_angles(
                hdrs,
                tuple(preset["add"]),
                tuple(preset["sub"]),
                {"INSTANGL": float(preset["instangl_default"])},
            )
            strategy = f"{preset['instrument']} preset"
            tried.append(strategy)

        if angles is None:
            for key in _DEFAULT_PA_KEYS:
                angles = _single_key_angles(hdrs, key)
                tried.append(f"keyword {key!r}")
                if angles is not None:
                    strategy = f"keyword {key!r}"
                    break
        if angles is None:
            angles = _combination_angles(
                hdrs,
                tuple(NIRC2_PRESET["add"]),
                tuple(NIRC2_PRESET["sub"]),
                {"INSTANGL": float(NIRC2_PRESET["instangl_default"])},
            )
            strategy = "PARANG + ROTPOSN - INSTANGL (Keck/NIRC2)"
            tried.append(strategy)
        if angles is None:
            lat = latitude if latitude is not None else None
            angles = _reconstructed_angles(hdrs, lat)
            strategy = "HA/DEC/latitude reconstruction"
            tried.append(strategy)

    if angles is None:
        available = sorted(
            str(k) for k in (getattr(hdrs[0], "keys", dict().keys)() or ())
        )
        raise ValueError(
            "could not obtain a parallactic angle for every frame. Strategies "
            f"tried: {tried}. Keywords present in the first header: "
            f"{available[:40]}{' ...' if len(available) > 40 else ''}. Pass "
            "`keys=...` (keyword, list of keywords or {'add':..., 'sub':...}) "
            "and/or `latitude=...` explicitly."
        )

    angles = np.asarray(angles, dtype=np.float64)
    if not np.all(np.isfinite(angles)):
        raise ValueError(
            f"the {strategy} yielded {int(np.sum(~np.isfinite(angles)))} "
            f"non-finite angle(s) out of {angles.size}."
        )
    unwrapped = _unwrap_degrees(angles)
    logger.info(
        "parallactic angles from %d header(s) via %s: %.3f -> %.3f deg "
        "(total field rotation %.3f deg).",
        len(hdrs), strategy, float(unwrapped[0]), float(unwrapped[-1]),
        float(unwrapped[-1] - unwrapped[0]),
    )
    return unwrapped


# --------------------------------------------------------------------------- #
# FITS I/O                                                                     #
# --------------------------------------------------------------------------- #
def load_fits(
    path: str | os.PathLike[str],
    ext: int | str = 0,
    return_header: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], Any]:
    """Read a FITS image or cube as a ``float64`` array.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the FITS file.
    ext : int or str, optional
        Extension index (0 = primary HDU) or ``EXTNAME``. Default 0.
    return_header : bool, optional
        If True, also return the header of the extension read.

    Returns
    -------
    numpy.ndarray or tuple
        The data as a C-contiguous float64 array (``(y, x)`` for an image,
        ``(n_frames, y, x)`` for a cube), or ``(data, header)`` when
        ``return_header`` is True.

    Raises
    ------
    ImportError
        If astropy is not installed.
    ValueError
        If the file does not exist, if the extension does not exist, or if the
        selected HDU holds no data (e.g. an empty primary HDU followed by an
        image extension -- pass ``ext=1`` in that case).

    Notes
    -----
    The file is opened with ``memmap=False`` and the array is copied, so the
    returned data stays valid after the file is closed and never aliases the
    on-disk buffer.  Integer FITS data (``BITPIX>0``) is promoted to float64,
    ``BZERO``/``BSCALE`` being applied by astropy beforehand.
    """
    fits = _import_fits()
    fname = _as_path(path)
    if not os.path.isfile(fname):
        raise ValueError(f"FITS file not found: {fname!r}.")

    with fits.open(fname, memmap=False) as hdul:
        try:
            hdu = hdul[ext]
        except (KeyError, IndexError) as exc:
            names = [
                f"{i}:{h.name!r}" for i, h in enumerate(hdul)
            ]
            raise ValueError(
                f"extension {ext!r} not found in {fname!r}; available "
                f"extensions: {names}."
            ) from exc
        data = hdu.data
        header = hdu.header.copy()
        if data is None:
            raise ValueError(
                f"extension {ext!r} of {fname!r} contains no data (shape "
                f"None). If this is an empty primary HDU, try ext=1."
            )
        out = np.array(data, dtype=np.float64, copy=True)

    logger.debug("loaded %r ext=%r: shape %s", fname, ext, out.shape)
    if return_header:
        return out, header
    return out


def save_fits(
    path: str | os.PathLike[str],
    data: ArrayLike,
    header: Any = None,
    overwrite: bool = True,
) -> None:
    """Write an array to a FITS file.

    Parameters
    ----------
    path : str or os.PathLike
        Output path. Missing parent directories are created.
    data : array_like
        Image ``(y, x)`` or cube ``(n_frames, y, x)``. Written as float32 if
        the input is float32, otherwise as float64 (integers are preserved).
        NaNs are kept as-is (FITS supports IEEE NaN).
    header : mapping or astropy.io.fits.Header, optional
        Cards to write. A plain dict is accepted; values may be
        ``(value, comment)`` tuples.
    overwrite : bool, optional
        Overwrite an existing file. Default True.

    Returns
    -------
    None

    Raises
    ------
    ImportError
        If astropy is not installed.
    ValueError
        If ``data`` is not a numeric array of 1--3 dimensions, or if the file
        exists and ``overwrite`` is False.

    Notes
    -----
    An ``EXOKLIP`` card holding the package version is added when absent, so
    that a reduced product can always be traced back to the code that made it.
    """
    fits = _import_fits()
    fname = _as_path(path)

    arr = np.asarray(data)
    if arr.dtype == np.dtype(object) or arr.dtype.kind not in "fiub":
        raise ValueError(
            f"`data` must be a numeric array, got dtype {arr.dtype!r}."
        )
    if arr.ndim not in (1, 2, 3):
        raise ValueError(
            f"`data` must be 1D, 2D (y, x) or 3D (n_frames, y, x), got shape "
            f"{arr.shape} with {arr.ndim} dimension(s)."
        )
    if arr.dtype.kind == "f" and arr.dtype != np.float32:
        arr = np.asarray(arr, dtype=np.float64)
    elif arr.dtype.kind == "b":
        arr = arr.astype(np.uint8)

    if os.path.exists(fname) and not overwrite:
        raise ValueError(
            f"{fname!r} already exists and `overwrite` is False."
        )
    parent = os.path.dirname(os.path.abspath(fname))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    hdr = fits.Header()
    if header is not None:
        if isinstance(header, fits.Header):
            hdr = header.copy()
        elif isinstance(header, Mapping):
            for key, value in header.items():
                hdr[str(key)] = value
        else:
            raise ValueError(
                f"`header` must be a mapping or an astropy Header, got "
                f"{type(header).__name__}."
            )
    if "EXOKLIP" not in hdr:
        try:
            from . import __version__ as _ver
        except ImportError:  # pragma: no cover
            _ver = "unknown"
        hdr["EXOKLIP"] = (str(_ver), "exoklip version that wrote this file")

    fits.PrimaryHDU(data=arr, header=hdr).writeto(fname, overwrite=overwrite)
    logger.info("wrote %s %s to %r", arr.shape, arr.dtype, fname)


def load_cube_from_dir(
    pattern: str | os.PathLike[str],
    sort_by_header: str | None = None,
    ext: int | str = 0,
) -> tuple[NDArray[np.float64], list[Any]]:
    """Build a cube from every FITS file matching a glob pattern.

    Parameters
    ----------
    pattern : str or os.PathLike
        Glob pattern (e.g. ``'data/raw/*.fits'``). A directory is also
        accepted and expanded to ``<dir>/*.fits``.
    sort_by_header : str, optional
        Header keyword used to sort the frames (e.g. ``'MJD-OBS'``,
        ``'UTC'``). Values are sorted numerically when they all parse as
        numbers, lexicographically otherwise. Default: sort by file path,
        which is the natural order for zero-padded sequential file names.
    ext : int or str, optional
        Extension read in every file. Default 0.

    Returns
    -------
    tuple
        ``(cube, headers)`` with ``cube`` of shape ``(n_frames, y, x)``
        (float64) and ``headers`` a list of length ``n_frames``.

    Raises
    ------
    ImportError
        If astropy is not installed.
    ValueError
        If the pattern matches no file, if a file has an incompatible frame
        shape, or if ``sort_by_header`` is missing from a header.

    Notes
    -----
    Files holding a 3D cube are unrolled into individual frames and their
    header is repeated for each plane, so ``len(headers) == cube.shape[0]``
    always holds -- which is what :func:`parallactic_angles_from_headers`
    expects.
    """
    pat = _as_path(pattern, "pattern")
    if os.path.isdir(pat):
        pat = os.path.join(pat, "*.fits")
    files = sorted(_glob.glob(pat))
    if not files:
        raise ValueError(
            f"no file matches the pattern {pat!r} (cwd: {os.getcwd()!r})."
        )

    frames: list[NDArray[np.float64]] = []
    headers: list[Any] = []
    for fname in files:
        data, header = load_fits(fname, ext=ext, return_header=True)
        if data.ndim == 2:
            frames.append(data)
            headers.append(header)
        elif data.ndim == 3:
            for plane in data:
                frames.append(plane)
                headers.append(header)
        else:
            raise ValueError(
                f"{fname!r} holds a {data.ndim}D array of shape {data.shape}; "
                f"expected a 2D image or a 3D cube."
            )

    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(
            f"the {len(frames)} frames read from {pat!r} do not share a common "
            f"shape; found {sorted(shapes)}. Crop them first "
            f"(exoklip.core.pad_or_crop)."
        )

    if sort_by_header is not None:
        raw_values = []
        for i, header in enumerate(headers):
            value = _header_get(header, sort_by_header, _MISSING)
            if value is _MISSING:
                raise ValueError(
                    f"keyword {sort_by_header!r} is missing from the header of "
                    f"frame {i} ({files[min(i, len(files) - 1)]!r})."
                )
            raw_values.append(value)
        try:
            sort_keys: list[Any] = [
                _to_float(v, f"{sort_by_header!r}") for v in raw_values
            ]
        except ValueError:
            sort_keys = [str(v) for v in raw_values]
        order = sorted(range(len(frames)), key=lambda i: sort_keys[i])
        frames = [frames[i] for i in order]
        headers = [headers[i] for i in order]
        logger.info("sorted %d frames by header keyword %r.",
                    len(frames), sort_by_header)

    cube = np.stack(frames, axis=0).astype(np.float64, copy=False)
    logger.info("loaded cube %s from %d file(s) matching %r.",
                cube.shape, len(files), pat)
    return cube, headers


# --------------------------------------------------------------------------- #
# native PNG decoder (zlib + struct only)                                      #
# --------------------------------------------------------------------------- #
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Number of samples per pixel for each PNG colour type.
_PNG_CHANNELS: dict[int, int] = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}

#: Bit depths allowed by the PNG specification for each colour type.
_PNG_DEPTHS: dict[int, tuple[int, ...]] = {
    0: (1, 2, 4, 8, 16),
    2: (8, 16),
    3: (1, 2, 4, 8),
    4: (8, 16),
    6: (8, 16),
}

_PNG_COLOUR_NAMES: dict[int, str] = {
    0: "greyscale", 2: "truecolour (RGB)", 3: "indexed (palette)",
    4: "greyscale+alpha", 6: "truecolour+alpha (RGBA)",
}


def _paeth_predictor(
    a: NDArray[np.int64], b: NDArray[np.int64], c: NDArray[np.int64]
) -> NDArray[np.int64]:
    """PNG Paeth predictor, vectorised over the ``bpp`` byte lanes.

    ``a`` is the byte to the left, ``b`` the byte above and ``c`` the byte
    above-left of the current one (PNG spec, sect. 9.4).
    """
    p = a + b - c
    pa = np.abs(p - a)
    pb = np.abs(p - b)
    pc = np.abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


def _png_defilter(
    raw: NDArray[np.uint8], height: int, stride: int, bpp: int
) -> NDArray[np.uint8]:
    """Undo the per-scanline PNG filters.

    Parameters
    ----------
    raw : numpy.ndarray
        ``(height, stride + 1)`` uint8 array: column 0 is the filter type of
        the scanline, columns ``1:`` are the filtered bytes.
    height, stride : int
        Number of scanlines and number of data bytes per scanline.
    bpp : int
        Number of bytes per complete pixel, rounded up to at least 1 -- the
        lag of the horizontal predictors.

    Returns
    -------
    numpy.ndarray
        ``(height, stride)`` uint8 array of reconstructed bytes.

    Raises
    ------
    ValueError
        If a scanline carries an unknown filter type.

    Notes
    -----
    Filters ``None``/``Sub``/``Up`` are fully vectorised (``Sub`` is a
    cumulative sum modulo 256 along each of the ``bpp`` byte lanes).  Filters
    ``Average`` and ``Paeth`` are *non-linear first-order recurrences over
    pixels* -- ``recon[i]`` depends on ``recon[i - bpp]`` through a floor
    division / a comparison -- so no closed form exists and a loop over pixel
    positions is mathematically unavoidable; the ``bpp`` byte lanes of each
    step are still handled as a vector.
    """
    n_groups = stride // bpp
    if n_groups * bpp != stride:  # pragma: no cover - guaranteed by caller
        raise ValueError(
            f"inconsistent PNG scanline: stride={stride} is not a multiple of "
            f"bpp={bpp}."
        )
    recon = np.zeros((height, stride), dtype=np.uint8)
    prior = np.zeros(stride, dtype=np.uint8)

    for row in range(height):
        ftype = int(raw[row, 0])
        cur = raw[row, 1:]
        if ftype == 0:  # None
            out = cur.copy()
        elif ftype == 1:  # Sub: recon[i] = cur[i] + recon[i - bpp]
            lanes = cur.reshape(n_groups, bpp).astype(np.int64)
            out = (np.cumsum(lanes, axis=0) & 0xFF).astype(np.uint8).reshape(stride)
        elif ftype == 2:  # Up: recon[i] = cur[i] + prior[i]  (uint8 wraps)
            out = (cur + prior).astype(np.uint8)
        elif ftype in (3, 4):  # Average / Paeth: sequential recurrence
            cur_l = cur.reshape(n_groups, bpp).astype(np.int64)
            pri_l = prior.reshape(n_groups, bpp).astype(np.int64)
            rec_l = np.empty((n_groups, bpp), dtype=np.int64)
            left = np.zeros(bpp, dtype=np.int64)        # recon[i - bpp]
            upleft = np.zeros(bpp, dtype=np.int64)      # prior[i - bpp]
            for k in range(n_groups):
                up = pri_l[k]
                if ftype == 3:
                    pred = (left + up) >> 1             # floor((a + b) / 2)
                else:
                    pred = _paeth_predictor(left, up, upleft)
                left = (cur_l[k] + pred) & 0xFF
                rec_l[k] = left
                upleft = up
            out = rec_l.astype(np.uint8).reshape(stride)
        else:
            raise ValueError(
                f"unknown PNG filter type {ftype} on scanline {row}; the "
                f"specification defines 0 (None), 1 (Sub), 2 (Up), "
                f"3 (Average) and 4 (Paeth)."
            )
        recon[row] = out
        prior = recon[row]
    return recon


def _png_unpack_samples(
    recon: NDArray[np.uint8],
    width: int,
    height: int,
    channels: int,
    bit_depth: int,
) -> NDArray[np.uint16]:
    """Turn reconstructed scanline bytes into ``(height, width, channels)``."""
    if bit_depth == 8:
        return recon.reshape(height, width, channels).astype(np.uint16)
    if bit_depth == 16:
        pairs = recon.reshape(height, width, channels, 2).astype(np.uint16)
        # Big-endian samples, spelled out to stay endianness-independent.
        return (pairs[..., 0] << 8) | pairs[..., 1]
    # Sub-byte depths (1, 2, 4): colour types 0 and 3 only, channels == 1.
    bits = np.unpackbits(recon, axis=1)
    needed = width * channels * bit_depth
    grouped = bits[:, :needed].reshape(height, width * channels, bit_depth)
    weights = (1 << np.arange(bit_depth - 1, -1, -1)).astype(np.uint16)
    values = (grouped.astype(np.uint16) * weights).sum(axis=2)
    return values.reshape(height, width, channels).astype(np.uint16)


def _decode_png(data: bytes) -> tuple[NDArray[np.uint16], int]:
    """Decode a PNG byte string with the standard library only.

    Supports the non-interlaced subset of the specification: colour types 0
    (greyscale), 2 (RGB), 3 (palette), 4 (greyscale+alpha) and 6 (RGBA), bit
    depths 1/2/4/8/16 where the specification allows them, and the five
    scanline filters.

    Parameters
    ----------
    data : bytes
        Whole file content.

    Returns
    -------
    tuple
        ``(samples, maxval)`` with ``samples`` a ``(height, width, channels)``
        uint16 array of raw sample values in file order (row 0 = **top** row)
        and ``maxval = 2**bit_depth - 1`` the full-scale value (255 for
        palette images, whose entries are 8-bit).

    Raises
    ------
    ValueError
        If the signature, the chunk structure, the colour type/bit depth
        combination or the scanline count is invalid, or if the image is
        Adam7-interlaced (unsupported).
    """
    if len(data) < 8 or data[:8] != _PNG_SIGNATURE:
        raise ValueError(
            "not a PNG file: the 8-byte signature "
            rf"b'\x89PNG\r\n\x1a\n' is missing (got {data[:8]!r})."
        )
    pos = 8
    ihdr: bytes | None = None
    plte: bytes | None = None
    idat: list[bytes] = []
    n_bytes = len(data)
    while pos + 8 <= n_bytes:
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if len(body) != length:
            raise ValueError(
                f"truncated PNG: chunk {ctype!r} at offset {pos} announces "
                f"{length} bytes but only {len(body)} are available."
            )
        crc_bytes = data[pos + 8 + length:pos + 12 + length]
        if len(crc_bytes) == 4:
            (crc,) = struct.unpack(">I", crc_bytes)
            if zlib.crc32(ctype + body) & 0xFFFFFFFF != crc:
                logger.warning(
                    "PNG chunk %r at offset %d has a bad CRC; decoding anyway.",
                    ctype.decode("ascii", "replace"), pos,
                )
        pos += 12 + length
        if ctype == b"IHDR":
            ihdr = body
        elif ctype == b"PLTE":
            plte = body
        elif ctype == b"IDAT":
            idat.append(body)
        elif ctype == b"IEND":
            break

    if ihdr is None or len(ihdr) != 13:
        raise ValueError(
            "invalid PNG: the 13-byte IHDR chunk is missing or malformed "
            f"(got {0 if ihdr is None else len(ihdr)} bytes)."
        )
    width, height, bit_depth, colour, compression, filter_method, interlace = (
        struct.unpack(">IIBBBBB", ihdr)
    )
    if width == 0 or height == 0:
        raise ValueError(f"invalid PNG dimensions {width}x{height}.")
    if compression != 0:
        raise ValueError(
            f"unsupported PNG compression method {compression}; only method 0 "
            f"(deflate) is defined."
        )
    if filter_method != 0:
        raise ValueError(
            f"unsupported PNG filter method {filter_method}; only method 0 is "
            f"defined."
        )
    if interlace != 0:
        raise ValueError(
            "Adam7-interlaced PNG files are not supported by the exoklip "
            "decoder. Re-save the file without interlacing, or convert it to "
            "FITS (which is what the science actually needs)."
        )
    if colour not in _PNG_CHANNELS:
        raise ValueError(
            f"unknown PNG colour type {colour}; expected one of "
            f"{sorted(_PNG_CHANNELS)}."
        )
    if bit_depth not in _PNG_DEPTHS[colour]:
        raise ValueError(
            f"bit depth {bit_depth} is not allowed for colour type {colour} "
            f"({_PNG_COLOUR_NAMES[colour]}); allowed depths: "
            f"{_PNG_DEPTHS[colour]}."
        )
    if not idat:
        raise ValueError("invalid PNG: no IDAT chunk found.")

    try:
        stream = zlib.decompress(b"".join(idat))
    except zlib.error as exc:
        raise ValueError(f"corrupted PNG: zlib could not inflate IDAT ({exc}).") from exc

    channels = _PNG_CHANNELS[colour]
    stride = (width * channels * bit_depth + 7) // 8
    bpp = max(1, (channels * bit_depth) // 8)
    expected = height * (stride + 1)
    if len(stream) != expected:
        raise ValueError(
            f"corrupted PNG: the inflated stream holds {len(stream)} bytes but "
            f"{height} scanlines of {stride} bytes (+1 filter byte) require "
            f"{expected}."
        )

    raw = np.frombuffer(stream, dtype=np.uint8).reshape(height, stride + 1)
    recon = _png_defilter(raw, height, stride, bpp)
    samples = _png_unpack_samples(recon, width, height, channels, bit_depth)

    maxval = (1 << bit_depth) - 1
    if colour == 3:
        if plte is None:
            raise ValueError(
                "invalid PNG: colour type 3 (indexed) requires a PLTE chunk."
            )
        palette = np.frombuffer(plte, dtype=np.uint8)
        if palette.size % 3 != 0:
            raise ValueError(
                f"invalid PNG palette: {palette.size} bytes is not a multiple "
                f"of 3."
            )
        palette = palette.reshape(-1, 3)
        indices = samples[..., 0]
        if int(indices.max()) >= palette.shape[0]:
            raise ValueError(
                f"invalid PNG: palette index {int(indices.max())} exceeds the "
                f"{palette.shape[0]} entries of the PLTE chunk."
            )
        samples = palette[indices].astype(np.uint16)
        maxval = 255

    logger.debug(
        "decoded PNG %dx%d, colour type %d (%s), bit depth %d.",
        width, height, colour, _PNG_COLOUR_NAMES[colour], bit_depth,
    )
    return samples, maxval


def _to_gray(samples: NDArray[Any], maxval: int) -> NDArray[np.float64]:
    """Normalise raw samples to ``[0, 1]`` float64 and drop colour/alpha.

    RGB is collapsed with an **unweighted** channel mean (not the BT.601 luma
    weights): for a grey source it is the linear combination that maximises the
    signal-to-noise ratio, and the display-oriented luma weights have no
    photometric meaning.  An alpha channel is discarded.
    """
    arr = np.asarray(samples, dtype=np.float64) / float(maxval)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(
            f"expected a 2D or 3D (y, x, channels) array, got shape "
            f"{arr.shape}."
        )
    n_channels = arr.shape[2]
    if n_channels == 1:
        return arr[..., 0]
    if n_channels == 2:  # greyscale + alpha
        logger.debug("dropping the alpha channel of a greyscale+alpha image.")
        return arr[..., 0]
    if n_channels in (3, 4):
        if n_channels == 4:
            logger.debug("dropping the alpha channel of an RGBA image.")
        return arr[..., :3].mean(axis=2)
    raise ValueError(
        f"cannot convert a {n_channels}-channel image to greyscale; expected "
        f"1, 2, 3 or 4 channels."
    )


def load_image_legacy(
    path: str | os.PathLike[str],
    flip: bool = True,
    warn: bool = True,
) -> NDArray[np.float64]:
    """Load a PNG or JPEG preview image as a 2D greyscale array.

    Provided so that the legacy single-image path of the package
    (``legacy/b_fixed.py``) runs without imageio or scikit-image. PNG is decoded
    natively with :mod:`zlib` and :mod:`struct`; JPEG needs Pillow, which is
    optional.

    .. warning::

       **Preview images are not science data.** An 8-bit file has 256
       distinguishable levels. A planet at a contrast of 1e-4 to 1e-6 sits far
       below one quantisation step of the stellar halo, so it is destroyed by
       the encoding before any algorithm runs — and JPEG additionally applies
       lossy block compression that invents and removes structure at exactly
       the spatial scale of a PSF. Detecting a companion in such a file is not
       possible in principle, not merely difficult. Use
       :func:`load_fits` on the original data.

    Parameters
    ----------
    path : str or path-like
        Image file. The format is chosen from the extension, falling back to
        sniffing the PNG signature.
    flip : bool, default True
        Flip vertically on load. Image formats store row 0 at the **top**,
        whereas this package indexes images with ``y`` increasing **upwards**
        (the convention of :func:`exoklip.core.frame_center` and of every plot
        drawn with ``origin='lower'``). Leaving this on keeps a loaded preview
        oriented like a loaded FITS file. Set it to False if you want the raw
        file order.
    warn : bool, default True
        Emit the quantisation warning. Only silence it if you are deliberately
        working on previews for display purposes.

    Returns
    -------
    ndarray
        ``(y, x)`` float64 array normalised to ``[0, 1]``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format is unsupported or the file is malformed.
    ImportError
        For JPEG input when Pillow is not installed.
    """
    resolved = _as_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"no such image file: {resolved}")

    with open(resolved, "rb") as handle:
        data = handle.read()
    if not data:
        raise ValueError(f"{resolved} is empty.")

    extension = os.path.splitext(resolved)[1].lower()

    if warn:
        logger.warning(
            "Loading %s as a preview image. An 8-bit file holds 256 levels, so a "
            "companion at 1e-4 contrast or fainter is quantised away before any "
            "algorithm sees it. Preview images cannot be used for detection - "
            "fetch the original FITS and use exoklip.io.load_fits.",
            os.path.basename(resolved),
        )

    if data[:8] == _PNG_SIGNATURE:
        samples, maxval = _decode_png(data)
        image = _to_gray(samples, maxval)
    elif extension in (".jpg", ".jpeg", ".jpe", ".jfif") or data[:2] == b"\xff\xd8":
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "reading JPEG requires Pillow, which is an optional dependency: "
                "pip install Pillow. PNG needs no extra package, and FITS is what "
                "you actually want for science - see exoklip.io.load_fits."
            ) from exc
        with Image.open(resolved) as handle:
            array = np.asarray(handle, dtype=np.float64)
        image = _to_gray(array, 255)
    else:
        raise ValueError(
            f"unsupported image format for {resolved!r} (extension {extension!r}). "
            "This loader handles PNG natively and JPEG through Pillow; for "
            "scientific data use exoklip.io.load_fits."
        )

    if image.ndim != 2:
        raise ValueError(
            f"decoded image has shape {image.shape}; expected a 2D greyscale array."
        )
    return np.flipud(image).copy() if flip else image
