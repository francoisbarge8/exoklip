"""exoklip — direct-imaging exoplanet detection with KLIP and ADI.

A dependency-light implementation of the post-processing chain used to pull
planets out of the speckle noise of a coronagraphic image sequence: angular
differential imaging, Karhunen-Loeve Image Projection, small-sample detection
statistics, throughput-corrected contrast curves, and a Fourier-optics simulator
to test all of it without needing a telescope.

Hard dependencies are numpy and scipy only. ``astropy`` (FITS I/O),
``matplotlib`` (figures) and ``tqdm`` (progress bars) are optional and imported
lazily, so ``import exoklip`` works on a bare install.

Quickstart
----------
>>> from exoklip import SimConfig, simulate_adi_sequence, klip_adi, snr_map
>>> sim = simulate_adi_sequence(SimConfig(n_frames=40, size=121, seed=0))
>>> image = klip_adi(sim['cube'], sim['angles'], fwhm=sim['fwhm'], n_modes=15)
>>> snr = snr_map(image, sim['fwhm'])

See ``examples/demo_full.py`` for the complete chain, from simulation through
detection to a contrast curve.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import (
    adi,
    core,
    detect,
    inject,
    io,
    klip,
    metrics,
    preproc,
    psf,
    rotation,
    simulate,
)
from .adi import klip_adi, median_adi, optimize_n_modes, pca_adi
from .core import frame_center, n_resolution_elements
from .detect import characterize, detect_sources, negfc_flux
from .inject import companion_position, inject_companions_cube, remove_companion
from .klip import klip_annular, klip_basis, klip_fullframe, klip_residual
from .metrics import (
    aperture_flux,
    contrast_curve,
    noise_profile,
    significance_threshold,
    snr_map,
    snr_student,
    throughput,
)
from .psf import create_synthetic_psf, fwhm_lambda_over_d, normalize_psf
from .rotation import cube_collapse, cube_derotate, frame_rotate
from .simulate import SimConfig, simulate_adi_sequence

__all__ = [
    "__version__",
    # submodules
    "adi",
    "core",
    "detect",
    "inject",
    "io",
    "klip",
    "metrics",
    "preproc",
    "psf",
    "rotation",
    "simulate",
    # simulation
    "SimConfig",
    "simulate_adi_sequence",
    # reduction
    "median_adi",
    "pca_adi",
    "klip_adi",
    "optimize_n_modes",
    "klip_basis",
    "klip_residual",
    "klip_annular",
    "klip_fullframe",
    # statistics
    "snr_student",
    "snr_map",
    "noise_profile",
    "significance_threshold",
    "aperture_flux",
    "throughput",
    "contrast_curve",
    # detection
    "detect_sources",
    "negfc_flux",
    "characterize",
    # utilities
    "frame_center",
    "n_resolution_elements",
    "frame_rotate",
    "cube_derotate",
    "cube_collapse",
    "normalize_psf",
    "create_synthetic_psf",
    "fwhm_lambda_over_d",
    "inject_companions_cube",
    "remove_companion",
    "companion_position",
]
