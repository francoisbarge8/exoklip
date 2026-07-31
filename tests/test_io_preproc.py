"""I/O and preprocessing.

These two modules deal with the parts of a reduction that are easy to get
silently wrong: a parallactic angle with the wrong sign derotates the sequence
backwards, and a centring error of half a pixel smears the whole field along an
arc. Both failures produce plausible-looking images, so they are checked here
against hand computations and synthetic ground truth.
"""

import struct
import zlib

import numpy as np
import pytest

from exoklip import io, preproc
from exoklip.core import dist_grid, frame_center
from exoklip.psf import gaussian_2d


# --------------------------------------------------------------------------- #
# Module surface
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", [io, preproc])
def test_every_advertised_export_exists(module):
    """__all__ must not promise functions the module does not define.

    Regression test: both modules once advertised public functions that were
    never written, so `from exoklip.io import load_image_legacy` raised
    ImportError and preproc had no public API at all.
    """
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert not missing, f"{module.__name__}.__all__ promises missing names: {missing}"


# --------------------------------------------------------------------------- #
# Parallactic angles
# --------------------------------------------------------------------------- #
def test_parallactic_angle_on_the_meridian():
    """At transit the answer is 0 or 180 depending on which side of zenith.

    PA = atan2(sin H, cos(dec) tan(lat) - sin(dec) cos H). At H = 0 the
    numerator vanishes, so the result is 0 when the denominator is positive
    (the target culminates south of zenith) and 180 when it is negative (north
    of zenith). Keck sits at latitude 19.8283 deg.
    """
    keck = 19.8283
    assert float(io.parallactic_angle(0.0, 0.0, keck)) == pytest.approx(0.0, abs=1e-9)
    assert float(io.parallactic_angle(0.0, 40.0, keck)) == pytest.approx(180.0, abs=1e-9)


def test_parallactic_angle_is_antisymmetric_in_hour_angle():
    """East and west of the meridian must mirror each other exactly."""
    angles = io.parallactic_angle([-2.0, 0.0, 2.0], 10.0, 19.8283)
    assert angles[1] == pytest.approx(0.0, abs=1e-9)
    assert angles[0] == pytest.approx(-angles[2], abs=1e-9)
    assert angles[2] > 0.0


def test_parallactic_angle_matches_an_explicit_computation():
    ha, dec, lat = 1.5, -12.0, 19.8283
    h = np.deg2rad(ha * 15.0)
    d, phi = np.deg2rad(dec), np.deg2rad(lat)
    expected = np.rad2deg(np.arctan2(np.sin(h), np.cos(d) * np.tan(phi) - np.sin(d) * np.cos(h)))
    assert float(io.parallactic_angle(ha, dec, lat)) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# Native PNG decoder
# --------------------------------------------------------------------------- #
def _encode_png_gray8(array: np.ndarray) -> bytes:
    """Minimal 8-bit greyscale PNG writer, so the decoder is tested against
    bytes this test file produced rather than against itself."""
    height, width = array.shape
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_png_round_trip(tmp_path):
    original = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / "ramp.png"
    path.write_bytes(_encode_png_gray8(original))

    decoded = io.load_image_legacy(path, warn=False, flip=False)
    assert np.allclose(decoded * 255.0, original)


def test_png_is_flipped_into_the_astronomical_orientation(tmp_path):
    """Image formats store row 0 at the top; this package indexes y upwards."""
    original = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / "ramp.png"
    path.write_bytes(_encode_png_gray8(original))

    assert np.allclose(
        io.load_image_legacy(path, warn=False, flip=True) * 255.0, np.flipud(original)
    )


def test_legacy_loader_rejects_unknown_formats(tmp_path):
    path = tmp_path / "not_an_image.dat"
    path.write_bytes(b"just some bytes")
    with pytest.raises(ValueError, match="unsupported image format"):
        io.load_image_legacy(path, warn=False)


def test_legacy_loader_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.load_image_legacy(tmp_path / "absent.png", warn=False)


def test_fits_functions_fail_with_an_actionable_message():
    """astropy is optional, so its absence must not leak a raw import error."""
    pytest.importorskip  # noqa: B018 - documents intent; astropy may be present
    try:
        import astropy.io.fits  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("astropy is installed; the fallback path cannot be exercised")

    with pytest.raises(ImportError, match="astropy"):
        io.load_fits("nonexistent.fits")


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
STAR_CENTER = (52.3, 48.7)
FWHM = 5.0


@pytest.fixture
def star_frame():
    base = gaussian_2d((101, 101), FWHM, center=STAR_CENTER, amplitude=1000.0) + 50.0
    return base + np.random.default_rng(0).normal(scale=1.0, size=(101, 101))


HOT_PIXELS = [(20, 30), (70, 80), (15, 90)]


def test_bad_pixel_correction_removes_hot_pixels(star_frame):
    corrupted = star_frame.copy()
    for y, x in HOT_PIXELS:
        corrupted[y, x] += 5000.0
    # Without protect_mask the stellar core is flagged too, which the function
    # warns about; that behaviour has its own test below.
    with pytest.warns(RuntimeWarning, match="brightest pixel"):
        cleaned = preproc.bad_pixel_correction(corrupted, sigma=5.0)
    for y, x in HOT_PIXELS:
        assert abs(cleaned[y, x] - star_frame[y, x]) < 100.0


def test_protect_mask_preserves_the_stellar_core(star_frame):
    """Without protection the core is flattened — it is a huge real outlier."""
    corrupted = star_frame.copy()
    for y, x in HOT_PIXELS:
        corrupted[y, x] += 5000.0
    protect = dist_grid((101, 101), center=STAR_CENTER) < 3 * FWHM
    cleaned = preproc.bad_pixel_correction(corrupted, sigma=5.0, protect_mask=protect)

    assert cleaned[52, 49] == pytest.approx(star_frame[52, 49], rel=1e-9)
    for y, x in HOT_PIXELS:
        assert abs(cleaned[y, x] - star_frame[y, x]) < 100.0


def test_bad_pixel_correction_warns_before_eating_the_brightest_pixel(star_frame):
    with pytest.warns(RuntimeWarning, match="brightest pixel"):
        preproc.bad_pixel_correction(star_frame, sigma=3.0)


@pytest.mark.parametrize("method", ["gaussian", "symmetry", "radon"])
def test_all_centring_methods_reach_sub_pixel_accuracy(star_frame, method):
    """Absil et al. 2013 require better than ~0.1 px for ADI.

    Regression test: the shift-based metrics take an *offset* from the
    geometric centre, and feeding them an absolute position used to consume the
    entire field budget, after which they silently returned the initial guess —
    an error of 70 px that looked like a successful call.
    """
    cy, cx = preproc.find_star_center(star_frame, FWHM, method=method)
    error = np.hypot(cy - STAR_CENTER[0], cx - STAR_CENTER[1])
    assert error < 0.1, f"{method} centring is off by {error:.3f} px"


def test_cube_recenter_recovers_known_shifts():
    rng = np.random.default_rng(1)
    offsets = (0.0, 2.0, -3.0)
    cube = np.array(
        [
            gaussian_2d((101, 101), FWHM, center=(50.0 + d, 50.0 - d), amplitude=1000.0)
            + rng.normal(scale=1.0, size=(101, 101))
            for d in offsets
        ]
    )
    _, shifts = preproc.cube_recenter(cube, FWHM, method="symmetry")
    for shift, d in zip(shifts, offsets):
        assert shift[0] == pytest.approx(-d, abs=0.1)
        assert shift[1] == pytest.approx(d, abs=0.1)


def test_frame_selection_keeps_the_quiet_frames_in_temporal_order():
    rng = np.random.default_rng(2)
    base = gaussian_2d((101, 101), FWHM, center=(50.0, 50.0), amplitude=1000.0)
    cube = np.array(
        [base + rng.normal(scale=s, size=(101, 101)) for s in (1.0, 60.0, 1.0, 90.0, 1.0)]
    )
    selected, indices = preproc.frame_selection(cube, FWHM, metric="corr", percentile=60.0)
    assert indices.tolist() == [0, 2, 4]
    assert selected.shape[0] == 3
    assert np.all(np.diff(indices) > 0), (
        "indices must stay sorted, otherwise slicing the angle array with them "
        "silently pairs each frame with the wrong parallactic angle"
    )


def test_temporal_binning_averages_frames_and_angles():
    cube = np.arange(5 * 4 * 4, dtype=float).reshape(5, 4, 4)
    angles = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    binned, binned_angles = preproc.temporal_binning(cube, angles, 2)

    assert binned.shape[0] == 3
    assert binned_angles.tolist() == [5.0, 25.0, 40.0]
    assert np.allclose(binned[0], cube[:2].mean(axis=0))
    assert np.allclose(binned[-1], cube[4]), "the trailing partial bin must not be padded"


def test_temporal_binning_rejects_mismatched_angles():
    cube = np.zeros((5, 4, 4))
    with pytest.raises(ValueError, match="exactly one parallactic angle per frame"):
        preproc.temporal_binning(cube, np.zeros(4), 2)


def test_subtract_background_zeroes_the_pedestal(star_frame):
    result = preproc.subtract_background(star_frame, method="median_annulus")
    outer = dist_grid((101, 101), center=frame_center((101, 101))) > 35
    assert abs(float(np.median(result[outer]))) < 0.5


def test_subtract_background_removes_a_tilted_plane():
    yy, xx = np.indices((61, 61), dtype=float)
    frame = 3.0 + 0.5 * yy - 0.25 * xx
    result = preproc.subtract_background(frame, method="plane")
    assert np.allclose(result, 0.0, atol=1e-9)
