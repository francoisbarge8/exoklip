"""End-to-end science: injection, simulation, statistics, and detection.

The acceptance test at the bottom is the one that matters. Everything else in
the package can be individually correct while the pipeline still fails to find a
planet, so this file checks the only claim a user actually cares about: a
companion invisible in the raw data is recovered, at the right place, with the
right brightness, and nothing else is.
"""

import numpy as np
import pytest

from exoklip import inject, metrics
from exoklip.adi import median_adi, pca_adi
from exoklip.core import frame_center
from exoklip.inject import companion_position
from exoklip.psf import create_synthetic_psf
from exoklip.rotation import cube_derotate
from exoklip.simulate import SimConfig, simulate_adi_sequence


# --------------------------------------------------------------------------- #
# Injection: the sign conventions, end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pa", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
def test_injected_companion_comes_back_at_the_requested_angle(pa):
    """Inject at PA, derotate, combine — the source must land at PA.

    This is the test that catches a sign error between injection and
    derotation. Such an error leaves every unit test green while smearing the
    companion into an arc and destroying the measured throughput.
    """
    rng = np.random.default_rng(0)
    shape = (30, 101, 101)
    cube = rng.normal(scale=0.01, size=shape)
    angles = np.linspace(-40.0, 40.0, shape[0])
    template = create_synthetic_psf(31, 4.0)

    injected = inject.inject_companions_cube(
        cube, template, angles, radius=25.0, pa=pa, flux=100.0
    )
    combined = np.nanmedian(cube_derotate(injected, angles), axis=0)

    center = frame_center(cube.shape)
    expected = companion_position(25.0, pa, center)
    peak = np.unravel_index(np.nanargmax(combined), combined.shape)
    offset = np.hypot(peak[0] - expected[0], peak[1] - expected[1])
    assert offset < 1.0, f"companion injected at PA={pa} recovered {offset:.2f} px away"


def test_remove_companion_cancels_inject_companion():
    rng = np.random.default_rng(1)
    cube = rng.normal(size=(10, 61, 61))
    angles = np.linspace(-20.0, 20.0, 10)
    template = create_synthetic_psf(21, 4.0)
    added = inject.inject_companions_cube(
        cube, template, angles, radius=15.0, pa=33.0, flux=50.0
    )
    restored = inject.remove_companion(
        added, template, angles, radius=15.0, pa=33.0, flux=50.0
    )
    assert np.allclose(restored, cube, atol=1e-9)


# --------------------------------------------------------------------------- #
# Simulator physics
# --------------------------------------------------------------------------- #
def test_speckles_are_pinned_to_the_detector():
    """A static wavefront error must freeze the speckle field in place.

    This is the property the whole technique rests on: if the speckles moved
    like the companion does, no amount of PSF subtraction could tell them
    apart. The comparison is made at *equal* turbulent power, so what is being
    measured is the effect of the static component alone.
    """
    from exoklip.core import get_annulus_mask

    # Correlate the speckle halo only. Including the stellar core would return
    # 0.99 for any pair of frames whatsoever — the core dominates the sum of
    # squares and says nothing about the speckles.
    halo = get_annulus_mask((101, 101), 10.0, 45.0)

    def correlation(cube):
        a, b = cube[0][halo], cube[-1][halo]
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    common = dict(
        n_frames=6, size=101, n_planets=0, planet_separations=(), planet_pas=(),
        planet_contrasts=(), photon_noise=False, read_noise=0.0, seed=2,
    )
    frozen = simulate_adi_sequence(
        SimConfig(static_phase_rms=0.8, dynamic_phase_rms=0.0, static_drift=0.0, **common)
    )
    mixed = simulate_adi_sequence(
        SimConfig(static_phase_rms=0.8, dynamic_phase_rms=0.25, static_drift=0.0, **common)
    )
    turbulent = simulate_adi_sequence(
        SimConfig(static_phase_rms=0.0, dynamic_phase_rms=0.25, **common)
    )

    assert correlation(frozen["cube"]) == pytest.approx(1.0, abs=1e-6), (
        "with no turbulent term and no drift the speckle field must be "
        "identical from frame to frame"
    )
    assert correlation(mixed["cube"]) > correlation(turbulent["cube"]) + 0.15, (
        "at equal turbulent power, adding a static wavefront error must raise "
        "the frame-to-frame correlation of the halo"
    )


def test_injected_contrast_is_exact():
    """The requested contrast must be what a photometric measurement returns."""
    common = dict(
        n_frames=2, size=101, static_phase_rms=0.0, dynamic_phase_rms=0.0,
        photon_noise=False, read_noise=0.0, pa_start=0.0, pa_end=0.0, seed=3,
    )
    with_planet = simulate_adi_sequence(
        SimConfig(planet_separations=(25.0,), planet_pas=(30.0,),
                  planet_contrasts=(1e-3,), **common)
    )
    without = simulate_adi_sequence(
        SimConfig(n_planets=0, planet_separations=(), planet_pas=(),
                  planet_contrasts=(), **common)
    )

    center = frame_center((101, 101))
    fwhm = with_planet["fwhm"]
    y, x = companion_position(25.0, 30.0, center)
    # Subtract the un-injected frame: at 25 px the stellar Airy wings still
    # contribute about 10 % of the companion flux to the same aperture.
    difference = with_planet["cube"][0] - without["cube"][0]
    measured = metrics.aperture_flux(difference, y, x, fwhm / 2.0)
    assert measured == pytest.approx(1e-3 * with_planet["star_flux"], rel=0.02)


def test_simulation_is_reproducible():
    a = simulate_adi_sequence(SimConfig(n_frames=4, size=101, seed=7))
    b = simulate_adi_sequence(SimConfig(n_frames=4, size=101, seed=7))
    assert np.array_equal(a["cube"], b["cube"])


# --------------------------------------------------------------------------- #
# Detection statistics
# --------------------------------------------------------------------------- #
def test_aperture_flux_on_a_constant_image_is_the_aperture_area():
    frame = np.ones((81, 81))
    assert metrics.aperture_flux(frame, 40.0, 40.0, 7.3) == pytest.approx(
        np.pi * 7.3**2, rel=0.005
    )


def test_snr_on_pure_noise_is_standard_normal():
    """The statistic must be calibrated: mean 0, unit spread, on noise alone."""
    rng = np.random.default_rng(11)
    values = []
    for _ in range(40):
        frame = rng.normal(size=(101, 101))
        for radius, pa in ((20.0, 33.0), (30.0, 77.0), (40.0, 201.0)):
            y, x = companion_position(radius, pa, frame_center(frame.shape))
            values.append(metrics.snr_student(frame, y, x, 4.0))
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    # With n samples the standard error on the mean is 1/sqrt(n).
    tolerance = 4.0 / np.sqrt(values.size)
    assert abs(values.mean()) < tolerance
    assert 0.7 < values.std() < 1.4


def test_small_sample_penalty_grows_as_the_separation_shrinks():
    """Mawet et al. 2014: the 5-sigma threshold blows up with few apertures."""
    far = metrics.significance_threshold(5.0, 60)
    mid = metrics.significance_threshold(5.0, 20)
    near = metrics.significance_threshold(5.0, 10)
    assert far < mid < near
    assert far == pytest.approx(5.0, abs=1.0), "far out it must tend to the Gaussian value"
    assert near > 10.0, "at 10 resolution elements the penalty must be severe"


def test_pixel_noise_underestimates_point_source_noise():
    """The classic optimism: per-pixel scatter ignores correlation within a PSF."""
    rng = np.random.default_rng(12)
    frame = rng.normal(size=(121, 121))
    aperture = metrics.noise_profile(frame, 4.0, method="aperture")
    pixel = metrics.noise_profile(frame, 4.0, method="pixel")
    i = len(aperture["radius"]) // 2
    assert aperture["noise"][i] > 3.0 * pixel["noise"][i]


# --------------------------------------------------------------------------- #
# Acceptance
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_klip_recovers_companions_that_the_raw_data_hides():
    """The end-to-end claim of the package.

    Two companions at 5e-3 and 1e-3 contrast, buried in a drifting speckle
    field. Neither is visible in a plain median of the sequence. After an
    annular KLIP reduction both must exceed a signal-to-noise ratio of 8, no
    other position may exceed 5, and KLIP must beat classical ADI on both.

    Those contrasts are relative to the *realised* stellar aperture flux, which
    is what ``SimConfig.star_flux`` now means. They were 2e-3 and 5e-4 while the
    simulator normalised on the diffraction-limited PSF — the same photons,
    labelled about 2.4 times too optimistically because the aberrated star had
    lost that much of its core flux to the halo.
    """
    sim = simulate_adi_sequence(
        SimConfig(
            n_frames=60, size=141, fwhm=4.0, n_planets=2,
            planet_separations=(20.0, 36.0), planet_pas=(60.0, 215.0),
            planet_contrasts=(5e-3, 1e-3), pa_start=-45.0, pa_end=45.0, seed=11,
        )
    )
    cube, angles, fwhm = sim["cube"], sim["angles"], sim["fwhm"]
    center = frame_center(cube.shape)

    klip_image = pca_adi(cube, angles, fwhm, n_modes=20, mode="annular", delta_rot=0.5)
    cadi_image = median_adi(cube, angles)
    raw_image = np.median(cube, axis=0)

    for truth in sim["truth"]:
        y, x = companion_position(truth["radius"], truth["pa"], center)
        snr_klip = metrics.snr_student(klip_image, y, x, fwhm, center=center)
        snr_cadi = metrics.snr_student(cadi_image, y, x, fwhm, center=center)
        snr_raw = metrics.snr_student(raw_image, y, x, fwhm, center=center)

        assert snr_klip > 8.0, (
            f"companion at r={truth['radius']} px, contrast {truth['contrast']:.0e} "
            f"recovered at only SNR {snr_klip:.2f}"
        )
        assert snr_klip > snr_cadi, "KLIP must beat classical ADI on a drifting field"
        assert snr_raw < 3.0, "the companion must be invisible before reduction"

    snr = metrics.snr_map(klip_image, fwhm, center=center, r_min=10.0, r_max=55.0)
    yy, xx = np.indices(klip_image.shape)
    elsewhere = np.ones(klip_image.shape, dtype=bool)
    for truth in sim["truth"]:
        y, x = companion_position(truth["radius"], truth["pa"], center)
        elsewhere &= np.hypot(yy - y, xx - x) > 1.5 * fwhm
    worst = np.nanmax(np.where(elsewhere & np.isfinite(snr), snr, -np.inf))
    assert worst < 5.0, f"a spurious {worst:.2f}-sigma peak survives the reduction"


@pytest.mark.slow
def test_throughput_is_below_one_and_falls_at_small_separation():
    """Self-subtraction is real and must be measured, not assumed away."""
    from functools import partial

    sim = simulate_adi_sequence(
        SimConfig(n_frames=30, size=121, fwhm=4.0, n_planets=0,
                  planet_separations=(), planet_pas=(), planet_contrasts=(),
                  pa_start=-40.0, pa_end=40.0, seed=13)
    )
    reduction = partial(pca_adi, fwhm=sim["fwhm"], n_modes=10, mode="annular", delta_rot=1.0)
    result = metrics.throughput(
        sim["cube"], sim["angles"], sim["psf"], sim["fwhm"],
        radii=[12.0, 24.0, 40.0], reduction_fn=reduction,
        n_branches=2, injection_contrast=3e-3, star_flux=sim["star_flux"],
    )
    tput = result["throughput"]
    assert np.all(tput > 0.0) and np.all(tput <= 1.2)
    assert tput[0] < tput[-1], (
        "throughput must degrade towards the star, where the companion moves "
        "least between frames and is most absorbed by its own PSF model"
    )


# --------------------------------------------------------------------------- #
# Regression: the two calibration defects found by adversarial review
# --------------------------------------------------------------------------- #
def test_snr_map_agrees_with_the_exact_statistic():
    """The map must not be wider than snr_student on pure noise.

    ``snr_map`` resamples a convolved aperture-flux map at the reference
    positions. A smoothing interpolator removes variance from the reference
    sample, under-estimates the noise and therefore inflates every SNR — the
    dangerous direction. With bilinear resampling the map was 8 % too wide,
    which multiplies the tail false-alarm probability at 5 by about 2.5.
    """
    rng = np.random.default_rng(4)
    from_map, exact = [], []
    for _ in range(12):
        frame = rng.normal(size=(121, 121))
        smap = metrics.snr_map(frame, 4.0, r_min=10.0, r_max=45.0)
        ys, xs = np.nonzero(np.isfinite(smap))
        for i in rng.choice(ys.size, size=40, replace=False):
            value = metrics.snr_student(frame, float(ys[i]), float(xs[i]), 4.0)
            if np.isfinite(value):
                from_map.append(smap[ys[i], xs[i]])
                exact.append(value)
    from_map = np.asarray(from_map)
    exact = np.asarray(exact)

    ratio = from_map.std() / exact.std()
    assert 0.95 < ratio < 1.05, (
        f"snr_map is {100 * (ratio - 1):+.1f} % wider than snr_student on pure "
        "noise; the resampling of the aperture-flux map is biasing the noise "
        "estimate"
    )
    assert np.sqrt(np.mean((from_map - exact) ** 2)) < 0.15


def test_detection_threshold_is_on_the_same_scale_as_the_reported_snr():
    """``threshold_5sigma`` must be comparable with ``snr``, not to the raw ratio.

    ``snr_student``/``snr_map`` already divide by ``sqrt(1 + 1/n2)``, so their
    output is Student-t distributed and the threshold beside them must be the
    bare quantile. ``significance_threshold`` carries the extra factor because
    it applies to the raw aperture-flux ratio used by a contrast curve; using
    it here charged the small-sample penalty twice.
    """
    from scipy import stats

    from exoklip.core import n_resolution_elements
    from exoklip.detect import detect_sources
    from exoklip.metrics import _student_quantile, significance_threshold

    for n in (10, 20, 60):
        n2 = len(metrics._comparison_indices(n, True))
        expected = stats.t.ppf(1.0 - stats.norm.sf(5.0), n2 - 1)
        assert _student_quantile(5.0, n) == pytest.approx(expected, rel=1e-12)
        assert significance_threshold(5.0, n) == pytest.approx(
            expected * np.sqrt(1.0 + 1.0 / n2), rel=1e-12
        )

    # ... and the value detect_sources reports is the former, not the latter.
    frame = np.zeros((121, 121))
    yy, xx = np.indices(frame.shape)
    y0, x0 = companion_position(30.0, 45.0, frame_center(frame.shape))
    frame += 9.0 * np.exp(-0.5 * ((yy - y0) ** 2 + (xx - x0) ** 2) / 2.0**2)
    found = detect_sources(frame, 4.0, threshold=5.0)
    assert found, "the synthetic source must be detected"
    n = n_resolution_elements(found[0]["radius"], 4.0)
    assert found[0]["threshold_5sigma"] == pytest.approx(_student_quantile(5.0, n))
    assert found[0]["threshold_5sigma"] < significance_threshold(5.0, n)
