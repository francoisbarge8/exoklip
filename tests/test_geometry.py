"""Geometry, rotation and PSF: the conventions everything else depends on.

A sign or centring error here does not raise an exception, it quietly moves
every companion by a few pixels and biases every contrast. These tests pin the
conventions down numerically.
"""

import numpy as np
import pytest

from exoklip import core, psf, rotation


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [5, 6, 101, 200])
def test_frame_center_is_half_of_n_minus_one(n):
    cy, cx = core.frame_center((n, n))
    assert cy == pytest.approx((n - 1) / 2.0)
    assert cx == pytest.approx((n - 1) / 2.0)


def test_annulus_mask_area_matches_analytic_area():
    r_in, r_out = 10.0, 25.0
    mask = core.get_annulus_mask((101, 101), r_in, r_out)
    expected = np.pi * (r_out**2 - r_in**2)
    assert mask.sum() == pytest.approx(expected, rel=0.02), (
        "the pixel count of an annulus must match its continuous area; a "
        "mismatch means the radial comparison is off by half a pixel"
    )


def test_position_angle_convention_is_north_up_east_left():
    """PA = 0 points at +y (North), PA = 90 at -x (East)."""
    grid = core.angle_grid((101, 101), convention="pa")
    cy, cx = core.frame_center((101, 101))
    ci, cj = int(cy), int(cx)
    assert grid[ci + 10, cj] == pytest.approx(0.0, abs=1e-6)
    assert grid[ci, cj - 10] == pytest.approx(90.0, abs=1e-6)
    assert grid[ci - 10, cj] == pytest.approx(180.0, abs=1e-6)
    assert grid[ci, cj + 10] == pytest.approx(270.0, abs=1e-6)


def test_segment_mask_wraps_through_zero():
    wrapped = core.get_segment_mask((101, 101), 10, 30, 350.0, 10.0)
    plain = core.get_segment_mask((101, 101), 10, 30, 10.0, 30.0)
    assert wrapped.sum() > 0, "a segment spanning 350->10 deg must not be empty"
    assert wrapped.sum() == pytest.approx(plain.sum(), rel=0.1)


def test_n_resolution_elements():
    assert core.n_resolution_elements(10.0, 4.0) == 15  # floor(2*pi*10/4)
    assert core.n_resolution_elements(0.1, 4.0) >= 1


def test_azimuthal_positions_are_on_the_circle_and_evenly_spaced():
    center = core.frame_center((101, 101))
    pos = core.azimuthal_positions(30.0, 4.0, center)
    radii = np.hypot(pos[:, 0] - center[0], pos[:, 1] - center[1])
    assert np.allclose(radii, 30.0, atol=1e-9)
    angles = np.sort(np.arctan2(pos[:, 1] - center[1], pos[:, 0] - center[0]))
    steps = np.diff(angles)
    assert np.allclose(steps, steps[0], atol=1e-9)


# --------------------------------------------------------------------------- #
# rotation
# --------------------------------------------------------------------------- #
def _point_source(shape=(101, 101), radius=25.0, pa=0.0, fwhm=4.0):
    center = core.frame_center(shape)
    ang = np.deg2rad(pa)
    y = center[0] + radius * np.cos(ang)
    x = center[1] - radius * np.sin(ang)
    return psf.gaussian_2d(shape, fwhm, center=(y, x), amplitude=1.0), (y, x)


def test_rotation_moves_position_angle_the_right_way():
    """frame_rotate(+90) must take a source from PA=0 to PA=90."""
    frame, _ = _point_source(pa=0.0)
    rotated = rotation.frame_rotate(frame, 90.0, cval=0.0)
    _, expected = _point_source(pa=90.0)
    peak = np.unravel_index(np.nanargmax(rotated), rotated.shape)
    assert np.hypot(peak[0] - expected[0], peak[1] - expected[1]) < 1.5


def test_rotation_round_trip_is_the_identity():
    rng = np.random.default_rng(0)
    frame = ndimage_smooth(rng.normal(size=(81, 81)))
    back = rotation.frame_rotate(rotation.frame_rotate(frame, 37.0, cval=0.0), -37.0, cval=0.0)
    interior = np.zeros_like(frame, dtype=bool)
    interior[20:-20, 20:-20] = True
    rms = np.sqrt(np.nanmean((back - frame)[interior] ** 2))
    assert rms < 0.02 * np.std(frame[interior])


def test_rotation_conserves_flux():
    frame, _ = _point_source(radius=20.0)
    rotated = rotation.frame_rotate(frame, 42.0, cval=0.0)
    assert np.nansum(rotated) == pytest.approx(np.nansum(frame), rel=0.01)


def test_nan_does_not_spread_across_the_frame():
    """A single NaN must stay a small blob, not smear over the whole frame.

    Cubic spline interpolation has infinite support once a NaN enters the
    prefilter, so the implementation has to mask around it instead. ``cval=0``
    keeps the un-mapped corners out of the count.
    """
    frame = np.ones((81, 81))
    frame[40, 40] = np.nan
    rotated = rotation.frame_rotate(frame, 30.0, cval=0.0)
    n_nan = int(np.sum(~np.isfinite(rotated)))
    assert 0 < n_nan < 40, (
        f"{n_nan} NaN after rotating a frame containing exactly one: the "
        "interpolation is propagating NaN instead of masking around it"
    )


def test_cube_derotate_undoes_the_field_rotation():
    """A source fixed in sky coordinates lands at the same place in every frame."""
    angles = np.linspace(-30.0, 30.0, 7)
    frames = []
    for a in angles:
        f, _ = _point_source(pa=60.0 - (-1.0) * a)  # sign=-1 convention
        frames.append(f)
    derotated = rotation.cube_derotate(np.array(frames), angles)
    _, expected = _point_source(pa=60.0)
    combined = np.nanmedian(derotated, axis=0)
    peak = np.unravel_index(np.nanargmax(combined), combined.shape)
    assert np.hypot(peak[0] - expected[0], peak[1] - expected[1]) < 1.5


def ndimage_smooth(a):
    from scipy import ndimage

    return ndimage.gaussian_filter(a, 2.0)


# --------------------------------------------------------------------------- #
# psf
# --------------------------------------------------------------------------- #
def test_airy_first_zero_and_fwhm_match_diffraction_theory():
    v = np.linspace(0.01, 6.0, 200_000)
    amp = psf._airy_amplitude(v, 0.0)
    first_zero = v[np.argmin(np.abs(amp))] / np.pi
    assert first_zero == pytest.approx(1.2197, abs=1e-3)
    assert psf._airy_fwhm_factor(0.0) == pytest.approx(1.0290, abs=2e-4)


def test_central_obscuration_narrows_the_core():
    assert psf._airy_fwhm_factor(0.33) < psf._airy_fwhm_factor(0.0)


@pytest.mark.parametrize("model", ["airy", "gaussian", "moffat"])
def test_normalised_psf_has_unit_aperture_flux(model):
    template = psf.create_synthetic_psf(61, 4.5, model=model)
    cy, cx = core.frame_center(template.shape)
    flux = psf._aperture_sum(template, cy, cx, 4.5 / 2.0)
    assert flux == pytest.approx(1.0, abs=1e-6), (
        "every contrast in the package assumes the template carries unit flux "
        "in a FWHM-diameter aperture"
    )


def test_gaussian_fit_recovers_the_injected_parameters():
    truth = psf.gaussian_2d(61, fwhm=4.7, center=(31.3, 29.8), amplitude=100.0)
    fit = psf.fit_gaussian_psf(truth, box=21)
    assert fit["fwhm"] == pytest.approx(4.7, rel=0.02)
    assert fit["y"] == pytest.approx(31.3, abs=0.05)
    assert fit["x"] == pytest.approx(29.8, abs=0.05)


def test_normalize_psf_recentres_an_offset_template():
    raw = psf.airy_2d(61, fwhm=4.5, center=(28.4, 33.1))
    template, _, _ = psf.normalize_psf(raw, fwhm=4.5)
    fit = psf.fit_gaussian_psf(template, box=15)
    cy, cx = core.frame_center(template.shape)
    assert fit["y"] == pytest.approx(cy, abs=0.05)
    assert fit["x"] == pytest.approx(cx, abs=0.05)


def test_fwhm_lambda_over_d_on_keck_nirc2():
    # Ks band, 10 m primary, NIRC2 narrow camera.
    assert psf.fwhm_lambda_over_d(2.12e-6, 10.0, 0.009942) == pytest.approx(4.53, abs=0.01)
