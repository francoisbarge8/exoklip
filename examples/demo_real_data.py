"""Template for reducing a real Keck/NIRC2 dataset.

This is a documented skeleton, not a runnable example: it needs FITS files you
have to download yourself, and astropy to read them. Every path is a parameter —
nothing here points at a directory that exists on your machine.

    pip install astropy
    python examples/demo_real_data.py --data-dir /path/to/fits --psf /path/to/psf.fits

Where to get data
-----------------
The Keck Observatory Archive (https://koa.ipac.caltech.edu/) serves NIRC2 data
publicly after its proprietary period. Download the **FITS**, not the JPEG
previews: an 8-bit preview has 256 levels and cannot hold a 1e-5 contrast
signal, so no algorithm applied to it can succeed.

You need three things:
  1. the science sequence, taken in pupil-tracking (vertical angle) mode,
  2. an unsaturated, unocculted image of the star for photometric calibration,
  3. the exposure-time ratio and any neutral-density transmission between the
     two, because the contrast scale depends directly on them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from pathlib import Path

import numpy as np

# Keck/NIRC2 narrow camera
PIXEL_SCALE = 0.009942     # arcsec/pixel
WAVELENGTH_KS = 2.146e-6   # m
DIAMETER = 10.0            # m


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", required=True, help="directory of science FITS")
    parser.add_argument("--pattern", default="*.fits")
    parser.add_argument("--psf", required=True, help="unsaturated calibration frame")
    parser.add_argument("--exptime-ratio", type=float, default=1.0,
                        help="science exposure time / calibration exposure time")
    parser.add_argument("--nd-transmission", type=float, default=1.0,
                        help="neutral density transmission used for the calibration frame")
    parser.add_argument("--n-modes", type=int, default=20)
    parser.add_argument("--delta-rot", type=float, default=1.0)
    parser.add_argument("--crop", type=int, default=301)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        import astropy  # noqa: F401
    except ImportError:
        print("error: astropy is required to read FITS files.\n"
              "       pip install astropy   (or: pip install 'exoklip[fits]')",
              file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"error: no such directory: {data_dir}", file=sys.stderr)
        return 2

    from exoklip import contrast_curve, detect_sources, klip_adi, snr_map
    from exoklip.core import cube_crop, frame_center
    from exoklip.io import load_cube_from_dir, load_fits, parallactic_angles_from_headers
    from exoklip.preproc import bad_pixel_correction, cube_recenter, frame_selection
    from exoklip.psf import fwhm_lambda_over_d, normalize_psf

    # ---------------------------------------------------------------- 1. load
    cube, headers = load_cube_from_dir(str(data_dir / args.pattern))
    print(f"Loaded {cube.shape[0]} frames of {cube.shape[1]}x{cube.shape[2]}")

    # Parallactic angles. For NIRC2 the sky PA is PARANG + ROTPOSN - INSTANGL,
    # and it must be unwrapped through the 180/-180 discontinuity — a sequence
    # that crosses it will otherwise derotate half its frames backwards.
    angles = parallactic_angles_from_headers(headers, keys="NIRC2")
    print(f"Field rotation: {angles.max() - angles.min():.1f} deg")
    if angles.max() - angles.min() < 15:
        print("warning: less than 15 deg of rotation. ADI needs the companion to "
              "move by more than a FWHM; at small separation it will be "
              "self-subtracted almost entirely.")

    # ------------------------------------------------------------ 2. preprocess
    fwhm = fwhm_lambda_over_d(WAVELENGTH_KS, DIAMETER, PIXEL_SCALE)
    print(f"Diffraction FWHM: {fwhm:.2f} px")

    cube = bad_pixel_correction(cube, sigma=5.0)
    # 'radon' finds the centre from the diffraction spikes and the halo symmetry,
    # so it still works when the core is saturated — which it usually is.
    cube, shifts = cube_recenter(cube, fwhm=fwhm, method="radon")
    print(f"Recentring shifts: {np.abs(shifts).max():.2f} px maximum")

    cube, kept = frame_selection(cube, fwhm=fwhm, metric="corr", percentile=90.0)
    angles = np.asarray(angles)[kept]
    print(f"Kept {cube.shape[0]} frames after quality selection")

    cube = cube_crop(cube, args.crop)

    # --------------------------------------------------- 3. photometric scale
    psf_raw = load_fits(args.psf)
    psf, aperture_flux, fwhm_measured = normalize_psf(psf_raw, size=int(8 * fwhm) | 1)
    # The star's flux AS IT WOULD APPEAR in a science frame.
    star_flux = aperture_flux * args.exptime_ratio / args.nd_transmission
    print(f"Stellar aperture flux (scaled to science frames): {star_flux:.4e}")

    # ------------------------------------------------------------- 4. reduce
    image = klip_adi(cube, angles, fwhm=fwhm, n_modes=args.n_modes,
                     delta_rot=args.delta_rot, n_jobs=4)
    center = frame_center(cube.shape)
    snr = snr_map(image, fwhm, center=center, r_min=2 * fwhm, n_jobs=4)

    candidates = detect_sources(snr, fwhm, threshold=5.0, center=center,
                                r_min=2 * fwhm, image=image)
    print(f"\n{len(candidates)} candidate(s):")
    for i, c in enumerate(candidates, 1):
        print(f"{i:<3d} r = {c['radius'] * PIXEL_SCALE:.3f}\" ({c['radius']:.1f} px)  "
              f"PA = {c['pa']:.1f} deg  SNR = {c['snr']:.1f}  "
              f"(5-sigma needs {c['threshold_5sigma']:.1f})")
    if candidates:
        print("\nBefore believing any of these: check that the candidate is present "
              "in independent halves of the sequence, that it does not sit on a "
              "diffraction spike, and that it moves with the sky between epochs.")

    # -------------------------------------------------------- 5. contrast curve
    reduction = partial(klip_adi, fwhm=fwhm, n_modes=args.n_modes, delta_rot=args.delta_rot)
    curve = contrast_curve(cube, angles, psf, fwhm, star_flux, reduction,
                           sigma=5.0, n_branches=3)
    print(f"\n{'sep (\")':>9s}{'contrast':>12s}{'dmag':>8s}{'throughput':>12s}")
    for i, r in enumerate(curve["radius"]):
        print(f"{r * PIXEL_SCALE:9.3f}{curve['contrast'][i]:12.2e}"
              f"{curve['delta_mag'][i]:8.2f}{curve['throughput'][i]:12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
