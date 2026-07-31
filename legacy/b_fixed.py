"""
================================================================================
 b_fixed.py — the original prototype, corrected. / le prototype d'origine, corrigé.
================================================================================

EN — WHAT THIS IS, AND WHAT IT IS NOT

    This is a repaired version of `b.py`, which ran PCA on overlapping spatial
    patches of a single image and called the result KLIP. The repair fixes eight
    real implementation bugs (listed below), but it cannot fix the premise:

        *** PATCH-PCA ON A SINGLE IMAGE IS NOT KLIP AND CANNOT FIND A PLANET. ***

    KLIP needs a reference library — many exposures in which the companion has
    moved and the speckles have not. With one frame there is no such library and
    no field rotation, so nothing distinguishes a planet from a speckle: both are
    compact, bright, and locally atypical. A high reconstruction error marks
    "unusual texture", not "astrophysical point source". Run this on a real
    coronagraphic image and the detection map you get is a map of the speckle
    halo. That is exactly what the original produced.

    What the corrected method IS good for: finding cosmic rays, hot pixels,
    detector defects, and extended structure that departs from the local texture
    of a single frame. Those are legitimate uses.

    For planet detection use `exoklip.adi.klip_adi` on a rotating sequence.

FR — CE QUE C'EST, ET CE QUE CE N'EST PAS

    Version réparée de `b.py`, qui faisait une PCA sur des patchs spatiaux
    recouvrants d'une seule image en appelant ça du KLIP. La réparation corrige
    huit vrais bugs (listés plus bas), mais ne peut pas corriger le principe :

        *** UNE PCA SUR PATCHS D'UNE SEULE IMAGE N'EST PAS DU KLIP
            ET NE PEUT PAS TROUVER DE PLANÈTE. ***

    KLIP a besoin d'une bibliothèque de références — de nombreuses poses où le
    compagnon a bougé et les speckles non. Avec une seule image, il n'y a ni
    bibliothèque ni rotation de champ : rien ne distingue une planète d'un
    speckle, les deux étant compacts, brillants et localement atypiques. Une
    erreur de reconstruction élevée signale « texture inhabituelle », pas
    « source ponctuelle astrophysique ». Appliquée à une vraie image
    coronographique, cette méthode produit une carte du halo de speckles — c'est
    littéralement ce que donnait l'original.

    Ce à quoi la méthode corrigée SERT vraiment : rayons cosmiques, pixels
    chauds, défauts de détecteur, structures étendues s'écartant de la texture
    locale. Ce sont des usages légitimes.

    Pour détecter des planètes, utilisez `exoklip.adi.klip_adi`.

--------------------------------------------------------------------------------
 THE EIGHT BUGS FIXED / LES HUIT BUGS CORRIGÉS
--------------------------------------------------------------------------------
 1. The iteration loop refit PCA on unchanged patches, so all 10 iterations were
    identical. Now the PCA is refit while EXCLUDING currently-flagged patches, so
    the model converges to the background and anomalies stand out. Stops when the
    flagged set stops changing.
 2. `skimage.measure.label` applied to a float RGB array made every distinct
    float value its own region. Now the map is thresholded to a 2D BINARY mask
    first, then labelled with `scipy.ndimage.label`.
 3. Overlapping patches were written with `=`, so later patches overwrote
    earlier ones. Now the residual is accumulated and divided by an overlap map.
 4. The 95th-percentile threshold always flagged exactly 5 % of patches, planet
    or no planet. Now the threshold is in units of a robust sigma (MAD-based), so
    a source-free image yields no detection.
 5. `inverse_transform` was called per patch in a Python loop over ~259 000
    patches. Now fully vectorised.
 6. Three identical colour channels from a monochrome IR detector tripled the
    cost. Now collapsed to one plane.
 7. `resize(image, (512, 512))` destroyed the PSF sampling. Now crop, never
    resize.
 8. Depended on imageio, scikit-image and scikit-learn. Now numpy + scipy only
    (PCA via SVD).

 Plus one addition the original needed: the radial profile of the stellar halo is
 subtracted first. Without it the reconstruction error is dominated by the halo's
 steep radial gradient rather than by anything localised — which is precisely why
 the original output image was a picture of the speckle halo.

 Usage:
     python legacy/b_fixed.py --demo
     python legacy/b_fixed.py --image path/to/frame.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exoklip.core import dist_grid, frame_center  # noqa: E402

logger = logging.getLogger("b_fixed")


def extract_patches(image: np.ndarray, patch_size: int, stride: int = 1) -> np.ndarray:
    """All (patch_size, patch_size) patches, as a (n_patches, patch_size**2) matrix.

    Uses a strided view rather than a Python double loop: for a 512x512 image and
    4-pixel patches this is 259 081 patches, which is not something to iterate
    over in Python.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D; got shape {image.shape}.")
    windows = np.lib.stride_tricks.sliding_window_view(image, (patch_size, patch_size))
    windows = windows[::stride, ::stride]
    return windows.reshape(-1, patch_size * patch_size)


def subtract_radial_profile(image: np.ndarray, center=None) -> np.ndarray:
    """Remove the azimuthally-averaged profile.

    The stellar halo falls by orders of magnitude across the frame. Left in
    place, it dominates every patch's reconstruction error and the method
    measures the halo instead of the anomalies.
    """
    ctr = frame_center(image.shape) if center is None else center
    r = dist_grid(image.shape, center=ctr)
    r_int = r.astype(int)
    n_bins = r_int.max() + 1
    total = np.bincount(r_int.ravel(), weights=np.nan_to_num(image).ravel(), minlength=n_bins)
    count = np.bincount(r_int.ravel(), minlength=n_bins)
    profile = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return image - profile[r_int]


def robust_sigma(values: np.ndarray) -> float:
    """MAD-based standard deviation estimate, insensitive to the outliers sought."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(finite - np.median(finite))))


def patch_pca_anomaly(
    image: np.ndarray,
    patch_size: int = 5,
    n_components: int = 3,
    n_iterations: int = 5,
    n_sigma: float = 5.0,
    mask_radius: float = 0.0,
    remove_profile: bool = True,
) -> dict:
    """Iteratively-refit patch PCA, flagging patches the model cannot reproduce.

    Parameters
    ----------
    image : ndarray
        Single 2D frame.
    patch_size : int, default 5
        Patch side in pixels. Should be comparable to the FWHM.
    n_components : int, default 3
        Principal components retained as the "background" model.
    n_iterations : int, default 5
        Maximum refit iterations. Converges when the flagged set stabilises.
    n_sigma : float, default 5.0
        Detection threshold in robust sigmas of the reconstruction error.
    mask_radius : float, default 0.0
        Radius of a central region excluded from the fit (the saturated core).
    remove_profile : bool, default True
        Subtract the radial profile first. Essentially mandatory on real data.

    Returns
    -------
    dict
        ``error_map``, ``detections`` (list of dicts), ``threshold``, ``sigma``,
        ``n_iterations_used``, ``expected_false_positives``.
    """
    work = np.array(image, dtype=np.float64, copy=True)
    if work.ndim == 3:
        # Bug 6: a monochrome IR detector does not have three colour channels.
        logger.warning(
            "Input has %d channels; collapsing to one. An IR detector is "
            "monochrome — the channels are redundant.", work.shape[2]
        )
        work = work.mean(axis=2)
    if remove_profile:
        work = subtract_radial_profile(work)

    ctr = frame_center(work.shape)
    radial = dist_grid(work.shape, center=ctr)

    patches = extract_patches(work, patch_size)
    patches = patches - patches.mean(axis=1, keepdims=True)
    n_side = work.shape[0] - patch_size + 1
    idx = np.arange(patches.shape[0])
    patch_y, patch_x = idx // n_side, idx % n_side

    usable = np.ones(patches.shape[0], dtype=bool)
    if mask_radius > 0:
        centres_r = radial[patch_y + patch_size // 2, patch_x + patch_size // 2]
        usable &= centres_r > mask_radius

    flagged = np.zeros(patches.shape[0], dtype=bool)
    error = np.zeros(patches.shape[0])
    used = 0

    for iteration in range(n_iterations):
        used = iteration + 1
        # Bug 1: fit on the patches NOT currently flagged, so the model converges
        # to the background instead of being dragged towards the anomalies.
        train = patches[usable & ~flagged]
        if train.shape[0] < n_components + 1:
            logger.warning("Too few clean patches to fit; stopping at iteration %d.", used)
            break

        mean = train.mean(axis=0)
        # Bug 8: PCA by SVD, no scikit-learn.
        _, _, vt = np.linalg.svd(train - mean, full_matrices=False)
        basis = vt[:n_components]

        # Bug 5: vectorised for all patches at once, not one at a time.
        centred = patches - mean
        reconstructed = (centred @ basis.T) @ basis
        error = np.linalg.norm(centred - reconstructed, axis=1)

        # Bug 4: a threshold in robust sigmas, so "nothing found" is possible.
        sigma = robust_sigma(error[usable])
        median = float(np.median(error[usable]))
        threshold = median + n_sigma * sigma
        new_flagged = usable & (error > threshold)

        if np.array_equal(new_flagged, flagged):
            logger.info("Converged after %d iterations.", used)
            flagged = new_flagged
            break
        flagged = new_flagged

    sigma = robust_sigma(error[usable])
    median = float(np.median(error[usable]))
    threshold = median + n_sigma * sigma

    # Bug 3: accumulate overlapping patches and divide by the overlap count,
    # instead of letting the last patch written win.
    error_map = np.zeros(work.shape)
    counts = np.zeros(work.shape)
    for dy in range(patch_size):
        for dx in range(patch_size):
            error_map[patch_y + dy, patch_x + dx] += error
            counts[patch_y + dy, patch_x + dx] += 1.0
    error_map = np.divide(error_map, counts, out=np.zeros_like(error_map), where=counts > 0)

    # Bug 2: threshold to a BINARY mask, then label. Labelling floats directly
    # gives one region per distinct value.
    binary = np.zeros(work.shape, dtype=bool)
    binary[patch_y + patch_size // 2, patch_x + patch_size // 2] = flagged
    binary = ndimage.binary_dilation(binary, np.ones((3, 3)))
    labels, n_labels = ndimage.label(binary)

    detections = []
    for lab in range(1, n_labels + 1):
        sel = labels == lab
        area = int(sel.sum())
        if area < 3:
            continue
        ys, xs = np.nonzero(sel)
        weights = error_map[ys, xs]
        cy = float((ys * weights).sum() / weights.sum())
        cx = float((xs * weights).sum() / weights.sum())
        peak = float(error_map[sel].max())
        detections.append(
            {
                "y": cy,
                "x": cx,
                "area": area,
                "peak_error": peak,
                "significance": (peak - median) / sigma if sigma > 0 else np.inf,
                "radius": float(np.hypot(cy - ctr[0], cx - ctr[1])),
            }
        )
    detections.sort(key=lambda d: d["significance"], reverse=True)

    # Honest accounting: with a Gaussian error distribution, this is how many
    # patches would cross the threshold by chance alone.
    from scipy import stats

    expected_fp = float(usable.sum() * stats.norm.sf(n_sigma))

    return {
        "error_map": error_map,
        "detections": detections,
        "threshold": threshold,
        "sigma": sigma,
        "median": median,
        "n_iterations_used": used,
        "expected_false_positives": expected_fp,
    }


def _demo() -> int:
    """Two experiments: what this method is good at, and what it cannot do.

    Part 1 is the legitimate use case — localised defects on a smooth background.
    Part 2 is the use case the original script was written for, and it fails,
    which is the whole point of the banner at the top of this file.
    """
    from exoklip.inject import companion_position
    from exoklip.psf import gaussian_2d
    from exoklip.simulate import SimConfig, simulate_adi_sequence

    rng = np.random.default_rng(5)
    ok = True

    # ------------------------------------------------------------------ part 1
    print("=" * 74)
    print(" PART 1 — what patch-PCA IS good for: localised defects")
    print("=" * 74)
    background = ndimage.gaussian_filter(rng.normal(size=(201, 201)), 8.0) * 40.0 + 500.0
    clean = background + rng.normal(scale=3.0, size=(201, 201))
    defective = clean.copy()
    hot = [(60, 45), (130, 88), (95, 160)]
    for y, x in hot:
        defective[y, x] += 900.0                      # hot pixels
    defective[150:153, 30:70] += 700.0                # a cosmic ray track
    print(f"Injected {len(hot)} hot pixels and one cosmic ray track.\n")

    for label, frame in (("WITH defects", defective), ("WITHOUT defects", clean)):
        result = patch_pca_anomaly(frame, patch_size=5, n_components=3,
                                   n_iterations=5, n_sigma=6.0, remove_profile=False)
        found = result["detections"]
        print(f"--- {label} ---")
        print(f"    {len(found)} detection(s), "
              f"{result['expected_false_positives']:.2f} expected by chance")
        for d in found[:5]:
            match = min(np.hypot(d["y"] - y, d["x"] - x) for y, x in hot)
            tag = "  <-- hot pixel" if match < 3 else (
                "  <-- cosmic ray" if 148 < d["y"] < 155 else "")
            print(f"      y={d['y']:6.1f} x={d['x']:6.1f} area={d['area']:4d} "
                  f"significance={d['significance']:7.1f}{tag}")
        if label == "WITH defects" and len(found) < len(hot):
            ok = False
        if label == "WITHOUT defects" and len(found) > 2:
            ok = False
        print()

    # ------------------------------------------------------------------ part 2
    print("=" * 74)
    print(" PART 2 — what it CANNOT do: find a planet in a speckle halo")
    print("=" * 74)
    sim = simulate_adi_sequence(
        SimConfig(n_frames=2, size=201, fwhm=4.0, n_planets=1,
                  planet_separations=(45.0,), planet_pas=(60.0,),
                  planet_contrasts=(3e-2,), pa_start=0.0, pa_end=0.0, seed=5)
    )
    truth = companion_position(45.0, 60.0, frame_center((201, 201)))
    print(f"Injected a companion at y={truth[0]:.1f}, x={truth[1]:.1f}, at a contrast")
    print("of 3e-2 — three hundred times brighter than a real target, and still:\n")

    result = patch_pca_anomaly(sim["cube"][0], patch_size=5, n_components=3,
                               n_iterations=5, n_sigma=6.0, mask_radius=12.0)
    found = result["detections"]
    recovered = [d for d in found if np.hypot(d["y"] - truth[0], d["x"] - truth[1]) < 6]
    print(f"    {len(found)} 'detection(s)'. The three strongest:")
    for d in found[:3]:
        offset = np.hypot(d["y"] - truth[0], d["x"] - truth[1])
        tag = "  <-- the companion" if offset < 6 else "  <-- speckle residual"
        print(f"      y={d['y']:6.1f} x={d['x']:6.1f} area={d['area']:5d} "
              f"significance={d['significance']:7.1f}{tag}")
    print(f"\n    Companion among the detections: {'yes' if recovered else 'NO'}")
    print("    The largest 'source' is the speckle halo itself, thousands of")
    print("    pixels in area. With a single frame there is no way to tell a")
    print("    speckle from a planet — they look identical by construction.\n")

    print("=" * 74)
    print(" Patch-PCA on one image detects defects, not planets. For planets:")
    print("   exoklip.adi.klip_adi(cube, angles, fwhm=..., n_modes=...)")
    print("=" * 74)
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Corrected single-image patch-PCA anomaly detection. NOT KLIP.",
        epilog="For actual exoplanet detection use exoklip.adi.klip_adi.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true", help="run the built-in demonstration")
    group.add_argument("--image", help="path to a PNG (or FITS with astropy installed)")
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--n-sigma", type=float, default=5.0)
    parser.add_argument("--mask-radius", type=float, default=0.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.demo:
        return _demo()

    path = Path(args.image)
    if path.suffix.lower() in (".fits", ".fit"):
        from exoklip.io import load_fits

        image = load_fits(str(path))
    else:
        from exoklip.io import load_image_legacy

        image = load_image_legacy(str(path))

    result = patch_pca_anomaly(
        image, patch_size=args.patch_size, n_components=args.n_components,
        n_sigma=args.n_sigma, mask_radius=args.mask_radius,
    )
    print(f"{len(result['detections'])} detection(s), "
          f"{result['expected_false_positives']:.2f} expected by chance:")
    for d in result["detections"]:
        print(f"  y={d['y']:7.1f} x={d['x']:7.1f} r={d['radius']:6.1f} "
              f"area={d['area']:4d} significance={d['significance']:6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
