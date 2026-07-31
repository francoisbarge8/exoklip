"""Command-line interface: ``exoklip <command>`` or ``python -m exoklip``.

Everything works with ``.npy`` files as well as FITS, so the CLI is usable
without astropy installed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import __version__

logger = logging.getLogger("exoklip")


def _load_array(path: str) -> np.ndarray:
    """Load a ``.npy`` or FITS array, whichever the extension says."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: no such file: {p}")
    if p.suffix.lower() in (".fits", ".fit", ".fts"):
        from .io import load_fits

        return np.asarray(load_fits(str(p)), dtype=np.float64)
    return np.asarray(np.load(str(p)), dtype=np.float64)


def _cmd_simulate(args: argparse.Namespace) -> int:
    from .simulate import SimConfig, simulate_adi_sequence

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sim = simulate_adi_sequence(
        SimConfig(
            n_frames=args.n_frames,
            size=args.size,
            fwhm=args.fwhm,
            n_planets=1,
            planet_separations=(args.separation,),
            planet_pas=(args.pa,),
            planet_contrasts=(args.contrast,),
            seed=args.seed,
        )
    )
    np.save(out / "cube.npy", sim["cube"])
    np.save(out / "angles.npy", sim["angles"])
    np.save(out / "psf.npy", sim["psf"])
    print(f"wrote cube.npy {sim['cube'].shape}, angles.npy, psf.npy to {out}")
    print(f"fwhm = {sim['fwhm']:.3f} px, star_flux = {sim['star_flux']:.3e}")
    for t in sim["truth"]:
        print(f"truth: r={t['radius']:.1f} px  PA={t['pa']:.1f} deg  "
              f"contrast={t['contrast']:.2e}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .adi import klip_adi, median_adi
    from .core import frame_center
    from .detect import detect_sources
    from .metrics import snr_map

    cube = _load_array(args.cube)
    angles = _load_array(args.angles).ravel()
    if cube.ndim != 3:
        raise SystemExit(f"error: cube must be 3D, got shape {cube.shape}")
    if angles.size != cube.shape[0]:
        raise SystemExit(
            f"error: {angles.size} angles for {cube.shape[0]} frames — "
            "there must be exactly one parallactic angle per frame"
        )

    center = frame_center(cube.shape)
    image = (
        median_adi(cube, angles)
        if args.mode == "cadi"
        else klip_adi(cube, angles, fwhm=args.fwhm, n_modes=args.n_modes,
                      mode=args.mode, delta_rot=args.delta_rot)
    )
    snr = snr_map(image, args.fwhm, center=center, r_min=args.r_min, r_max=args.r_max)
    found = detect_sources(
        snr, args.fwhm, threshold=args.threshold, center=center,
        r_min=args.r_min, r_max=args.r_max, image=image,
    )

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "reduced.npy", image)
        np.save(out / "snr.npy", snr)
        print(f"wrote reduced.npy and snr.npy to {out}")

    if not found:
        print(f"No source above SNR {args.threshold:g}.")
        return 1
    print(f"{len(found)} candidate(s) above SNR {args.threshold:g}:")
    print(f"{'':3s}{'r (px)':>9s}{'PA (deg)':>10s}{'SNR':>8s}{'5-sig thr':>11s}")
    for i, c in enumerate(found, 1):
        print(f"{i:<3d}{c['radius']:9.2f}{c['pa']:10.2f}{c['snr']:8.2f}"
              f"{c['threshold_5sigma']:11.2f}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    demo = Path(__file__).resolve().parent.parent / "examples" / "demo_full.py"
    if not demo.exists():
        raise SystemExit(
            f"error: {demo} not found. The demo ships with the source tree; "
            "clone the repository to run it."
        )
    import runpy

    sys.argv = [str(demo)]
    runpy.run_path(str(demo), run_name="__main__")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exoklip",
        description="KLIP/ADI post-processing for direct imaging of exoplanets.",
    )
    parser.add_argument("--version", action="version", version=f"exoklip {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for INFO, -vv for DEBUG")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="write a synthetic ADI dataset to disk")
    p.add_argument("-o", "--output", default="exoklip_data")
    p.add_argument("--n-frames", type=int, default=40)
    p.add_argument("--size", type=int, default=141)
    p.add_argument("--fwhm", type=float, default=4.0)
    p.add_argument("--separation", type=float, default=28.0)
    p.add_argument("--pa", type=float, default=60.0)
    p.add_argument("--contrast", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=_cmd_simulate)

    p = sub.add_parser("run", help="reduce a sequence and list detections")
    p.add_argument("--cube", required=True, help=".npy or .fits cube (n_frames, y, x)")
    p.add_argument("--angles", required=True, help=".npy or .fits parallactic angles")
    p.add_argument("--fwhm", type=float, required=True)
    p.add_argument("--mode", choices=["annular", "fullframe", "cadi"], default="annular")
    p.add_argument("--n-modes", type=int, default=20)
    p.add_argument("--delta-rot", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=5.0)
    p.add_argument("--r-min", type=float, default=None)
    p.add_argument("--r-max", type=float, default=None)
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("demo", help="run the end-to-end demonstration")
    p.set_defaults(func=_cmd_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
