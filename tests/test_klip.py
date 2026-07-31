"""The KLIP linear algebra, checked against theory and an independent solver.

Soummer, Pueyo & Larkin (2012) define KLIP as a projection onto the leading
eigenvectors of the reference covariance. That definition makes four properties
mandatory, and they are what these tests assert: the basis is orthonormal, the
residual is orthogonal to the basis, a target lying in the span of the
references cancels exactly, and the residual energy decreases monotonically with
the number of modes retained.
"""

import numpy as np
import pytest

from exoklip import klip


@pytest.fixture
def references():
    return np.random.default_rng(0).normal(size=(12, 400))


@pytest.fixture
def target():
    return np.random.default_rng(1).normal(size=400)


def independent_klip(target, refs, n_modes):
    """A deliberately separate implementation, via SVD rather than eigh.

    Written out in full here on purpose: comparing the package against itself
    would prove nothing.
    """
    centred = refs - refs.mean(axis=1, keepdims=True)
    target_c = target - target.mean()
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    basis = vt[:n_modes]
    return target_c - (target_c @ basis.T) @ basis


def test_kl_basis_is_orthonormal(references):
    basis = klip.klip_basis(references, n_modes=8)
    gram = basis @ basis.T
    assert np.allclose(gram, np.eye(8), atol=1e-10), (
        "the KL modes are eigenvectors of a symmetric matrix and must be "
        "orthonormal; they are not, so the 1/sqrt(eigenvalue) normalisation is wrong"
    )


def test_residual_is_orthogonal_to_the_basis(references, target):
    basis = klip.klip_basis(references, n_modes=8)
    residual = klip.klip_project(target, basis, n_modes=8)
    assert np.abs(residual @ basis.T).max() < 1e-9


def test_target_inside_the_reference_span_cancels_exactly(references):
    """With as many modes as references, a linear combination must vanish."""
    centred = references - references.mean(axis=1, keepdims=True)
    coefficients = np.random.default_rng(2).normal(size=references.shape[0])
    target = coefficients @ centred
    residual = klip.klip_residual(target, references, n_modes=references.shape[0])
    assert np.linalg.norm(residual) / np.linalg.norm(target) < 1e-8


def test_residual_energy_decreases_with_more_modes(references, target):
    energies = [
        np.linalg.norm(klip.klip_residual(target, references, n_modes=k))
        for k in range(1, 12)
    ]
    assert all(
        energies[i] >= energies[i + 1] - 1e-12 for i in range(len(energies) - 1)
    ), f"residual energy must not increase with K; got {energies}"


def test_matches_an_independent_svd_implementation(references, target):
    ours = klip.klip_residual(target, references, n_modes=8)
    theirs = independent_klip(target, references, 8)
    assert np.allclose(ours, theirs, atol=1e-8)


def test_two_entry_points_agree(references, target):
    direct = klip.klip_residual(target, references, n_modes=8)
    staged = klip.klip_project(target, klip.klip_basis(references, 8), n_modes=8)
    assert np.allclose(direct, staged, atol=1e-12)


def test_rank_deficient_references_are_truncated_not_crashed():
    """Duplicate references make the Gram matrix singular; KLIP must cope."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(5, 200))
    refs = np.vstack([base, base])  # rank 5, ten rows
    basis = klip.klip_basis(refs, n_modes=10)
    assert basis.shape[0] <= 5
    assert np.allclose(basis @ basis.T, np.eye(basis.shape[0]), atol=1e-9)


def test_klip_beats_least_squares_nowhere_and_reaches_it_at_full_rank():
    """At K = n_ref the projection must equal the exact least-squares fit."""
    rng = np.random.default_rng(4)
    refs = rng.normal(size=(10, 300))
    target = rng.normal(size=300)
    residual = klip.klip_residual(target, refs, n_modes=10)

    centred = refs - refs.mean(axis=1, keepdims=True)
    target_c = target - target.mean()
    coef, *_ = np.linalg.lstsq(centred.T, target_c, rcond=None)
    best = target_c - centred.T @ coef
    assert np.linalg.norm(residual) == pytest.approx(np.linalg.norm(best), rel=1e-8)


def test_rotation_threshold_matches_a_hand_computation():
    """|dPA| * r * pi/180 > delta_rot * fwhm, evaluated by hand.

    At r = 25 px and fwhm = 4 px with delta_rot = 1, the threshold is
    4 / 25 * 180 / pi = 9.167 deg, so only the frame 10 deg away qualifies.
    """
    angles = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    mask = klip.rotation_threshold_mask(angles, index=0, radius=25.0, fwhm=4.0)
    assert mask.tolist() == [False, False, False, False, True]


def test_the_frame_is_never_its_own_reference():
    angles = np.linspace(0.0, 90.0, 10)
    for i in range(len(angles)):
        mask = klip.rotation_threshold_mask(angles, index=i, radius=30.0, fwhm=4.0)
        assert not mask[i]


def test_annular_reduction_leaves_nan_outside_the_processed_zones():
    rng = np.random.default_rng(5)
    cube = rng.normal(size=(10, 61, 61))
    angles = np.linspace(-20.0, 20.0, 10)
    residuals = klip.klip_annular(cube, angles, fwhm=4.0, n_modes=3, r_min=10.0, r_max=20.0)
    from exoklip.core import dist_grid, frame_center

    r = dist_grid((61, 61), center=frame_center((61, 61)))
    assert np.all(~np.isfinite(residuals[0][r < 5])), "the core must stay untouched (NaN)"
    assert np.any(np.isfinite(residuals[0][(r > 12) & (r < 18)]))


def test_mode_sweep_matches_individual_calls():
    """The {K: cube} path reuses one eigendecomposition; it must not change the answer."""
    rng = np.random.default_rng(6)
    cube = rng.normal(size=(12, 61, 61))
    angles = np.linspace(-25.0, 25.0, 12)
    swept = klip.klip_annular(cube, angles, fwhm=4.0, n_modes=[3, 6], r_min=8.0, r_max=24.0)
    for k in (3, 6):
        single = klip.klip_annular(
            cube, angles, fwhm=4.0, n_modes=k, r_min=8.0, r_max=24.0
        )
        assert np.allclose(swept[k], single, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("n_segments", [1, 4, "auto"])
def test_annular_zones_partition_the_field_exactly(n_segments):
    """Every pixel of the requested range is processed exactly once.

    A gap between annuli leaves untouched starlight that later reads as a
    detection; an overlap means some pixels are subtracted twice. Neither shows
    up in the reduced image as anything but plausible structure, so it is
    checked directly on the map of processed pixels.
    """
    from exoklip.core import dist_grid, frame_center

    rng = np.random.default_rng(0)
    cube = rng.normal(size=(12, 101, 101))
    angles = np.linspace(-25.0, 25.0, 12)

    residuals = klip.klip_annular(
        cube, angles, fwhm=4.0, n_modes=4, asize=2.0,
        n_segments=n_segments, r_min=8.0, r_max=40.0,
    )
    processed = np.isfinite(residuals[0])
    radial = dist_grid((101, 101), center=frame_center((101, 101)))
    requested = (radial >= 8.0) & (radial < 40.0)

    assert int((requested & ~processed).sum()) == 0, "gap between annuli"
    assert int((~requested & processed).sum()) == 0, "processing outside the range"


def test_annular_coverage_does_not_depend_on_the_segmentation():
    from exoklip.core import dist_grid, frame_center

    rng = np.random.default_rng(1)
    cube = rng.normal(size=(12, 101, 101))
    angles = np.linspace(-25.0, 25.0, 12)
    counts = {
        seg: int(
            np.isfinite(
                klip.klip_annular(
                    cube, angles, fwhm=4.0, n_modes=4, asize=2.0,
                    n_segments=seg, r_min=8.0, r_max=40.0,
                )[0]
            ).sum()
        )
        for seg in (1, 4, "auto")
    }
    assert len(set(counts.values())) == 1, f"segmentation changes coverage: {counts}"


def test_parallel_execution_is_bitwise_identical():
    """Threads must not change a single bit — otherwise there is a race."""
    from exoklip.metrics import snr_map

    rng = np.random.default_rng(2)
    cube = rng.normal(size=(12, 101, 101))
    angles = np.linspace(-25.0, 25.0, 12)

    serial = klip.klip_annular(cube, angles, fwhm=4.0, n_modes=4, r_min=8.0, r_max=40.0, n_jobs=1)
    threaded = klip.klip_annular(cube, angles, fwhm=4.0, n_modes=4, r_min=8.0, r_max=40.0, n_jobs=4)
    assert np.allclose(serial, threaded, equal_nan=True, rtol=0, atol=0)

    image = np.nanmedian(serial, axis=0)
    assert np.allclose(
        snr_map(image, 4.0, r_min=10.0, r_max=40.0, n_jobs=1),
        snr_map(image, 4.0, r_min=10.0, r_max=40.0, n_jobs=4),
        equal_nan=True, rtol=0, atol=0,
    )
