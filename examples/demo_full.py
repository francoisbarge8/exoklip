"""End-to-end demonstration on a simulated dataset.

Simulates a 60-frame ADI sequence containing two companions, reduces it three
different ways, detects what is there, and derives a throughput-corrected
contrast curve. Writes every figure to ``examples/output/``.

    python examples/demo_full.py

Runs in about a minute and needs nothing but numpy, scipy and matplotlib.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path

import numpy as np

from exoklip import (
    SimConfig,
    contrast_curve,
    detect_sources,
    klip_adi,
    median_adi,
    pca_adi,
    simulate_adi_sequence,
    snr_map,
    throughput,
)
from exoklip.core import frame_center
from exoklip.inject import companion_position
from exoklip.metrics import snr_student
from exoklip.plotting import (
    plot_adi_principle,
    plot_contrast_curve,
    plot_reduction_summary,
    plot_snr_map,
    plot_throughput,
)

OUTPUT = Path(__file__).resolve().parent / "output"
NIRC2_PIXEL_SCALE = 0.009942  # arcsec/pixel, Keck/NIRC2 narrow camera


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # ---------------------------------------------------------------- simulate
    config = SimConfig(
        n_frames=60,
        size=141,
        fwhm=4.0,
        n_planets=2,
        planet_separations=(20.0, 36.0),
        planet_pas=(60.0, 215.0),
        planet_contrasts=(2e-3, 5e-4),
        pa_start=-45.0,
        pa_end=45.0,
        seed=11,
    )
    sim = simulate_adi_sequence(config)
    cube, angles, fwhm = sim["cube"], sim["angles"], sim["fwhm"]
    center = frame_center(cube.shape)
    truth_xy = [
        companion_position(t["radius"], t["pa"], center) for t in sim["truth"]
    ]
    print(f"Simulated {cube.shape[0]} frames of {cube.shape[1]}x{cube.shape[2]}, "
          f"FWHM {fwhm:.2f} px, field rotation {angles[-1] - angles[0]:.0f} deg")

    # ----------------------------------------------------------------- reduce
    reductions = {
        "Raw median": np.median(cube, axis=0),
        "Classical ADI": median_adi(cube, angles),
        "KLIP full-frame (K=20)": pca_adi(cube, angles, fwhm, n_modes=20, mode="fullframe"),
        "KLIP annular (K=20)": klip_adi(cube, angles, fwhm, n_modes=20, delta_rot=0.5),
    }
    best = reductions["KLIP annular (K=20)"]

    print(f"\n{'reduction':26s} " + "  ".join(f"SNR#{i + 1}" for i in range(len(sim['truth']))))
    print("-" * 46)
    for name, image in reductions.items():
        values = [
            snr_student(image, y, x, fwhm, center=center) for y, x in truth_xy
        ]
        print(f"{name:26s} " + "  ".join(f"{v:6.2f}" for v in values))

    # ---------------------------------------------------------------- detect
    snr = snr_map(best, fwhm, center=center, r_min=10.0, r_max=55.0)
    candidates = detect_sources(
        snr, fwhm, threshold=5.0, center=center, r_min=10.0, r_max=55.0, image=best
    )
    print(f"\n{len(candidates)} candidate(s) above SNR 5:")
    print(f"{'':3s}{'r (px)':>8s}{'PA (deg)':>10s}{'SNR':>8s}{'5-sig thr':>11s}")
    for i, c in enumerate(candidates, 1):
        print(f"{i:<3d}{c['radius']:8.2f}{c['pa']:10.2f}{c['snr']:8.2f}"
              f"{c['threshold_5sigma']:11.2f}")
    print("\ntruth:")
    for t in sim["truth"]:
        print(f"   r={t['radius']:.1f} px  PA={t['pa']:.1f} deg  contrast={t['contrast']:.1e}")

    # ------------------------------------------------------- throughput/limits
    print("\nMeasuring throughput by fake-companion injection "
          "(this is the slow part) ...")
    reduction = partial(klip_adi, fwhm=fwhm, n_modes=20, delta_rot=0.5)
    radii = np.arange(8.0, 56.0, 8.0)
    tput = throughput(
        cube, angles, sim["psf"], fwhm, radii, reduction,
        n_branches=3, injection_contrast=3e-3, star_flux=sim["star_flux"],
    )
    curve = contrast_curve(
        cube, angles, sim["psf"], fwhm, sim["star_flux"], reduction,
        sigma=5.0, radii=radii, throughput_result=tput,
    )
    print(f"\n{'r (px)':>8s}{'throughput':>12s}{'5-sig contrast':>16s}{'penalty':>10s}")
    for i, r in enumerate(curve["radius"]):
        print(f"{r:8.1f}{curve['throughput'][i]:12.3f}"
              f"{curve['contrast'][i]:16.2e}{curve['sigma_corr'][i]:10.2f}")

    # ---------------------------------------------------------------- figures
    import matplotlib

    matplotlib.use("Agg")
    figures = {
        "adi_principle.png": plot_adi_principle(
            cube, angles, fwhm, 36.0, 215.0, indices=(0, -1)
        ),
        "reductions.png": plot_reduction_summary(reductions, fwhm, truth_xy),
        "snr_map.png": plot_snr_map(snr, fwhm, 5.0, truth_xy),
        "contrast_curve.png": plot_contrast_curve(curve, pixel_scale=NIRC2_PIXEL_SCALE),
        "throughput.png": plot_throughput(tput),
    }
    for name, fig in figures.items():
        path = OUTPUT / name
        fig.savefig(path, dpi=130)
        print(f"wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}"
              f"  ({path.stat().st_size // 1024} kB)")

    print(f"\nTotal runtime: {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
