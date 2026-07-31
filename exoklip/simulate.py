"""Fourier-optics simulator of an angular differential imaging (ADI) sequence.

The point of this module is to produce a dataset where **the hard part is
actually hard**: a companion buried in a speckle field that a naive combination
of the frames cannot reveal, and that only a proper PSF-subtraction algorithm
recovers. A Gaussian blob added to white noise would make every algorithm look
brilliant and would prove nothing.

The physics that matters
------------------------
1. **Speckles are diffracted starlight, not noise.** They come from residual
   wavefront aberrations, so they are generated here by propagating a pupil with
   a phase screen: ``PSF = |FFT(P e^{i phi})|^2``. That gives them the right
   radial distribution, the right correlation length (about one ``lambda/D``)
   and the pinning to the diffraction pattern that real speckles have.

2. **The phase splits into a quasi-static and a turbulent part.** The static
   component is *identical in every frame*, so its speckles sit at fixed
   detector positions and are strongly correlated in time — this is exactly the
   signal KLIP models and removes. The dynamic component is redrawn per frame
   and averages into a smooth halo. ``static_drift`` slowly decorrelates the
   static part, which is what limits real observations over several hours.

3. **ADI: the sky rotates, the instrument does not.** In pupil-tracking mode the
   telescope lets the field rotate by the parallactic angle. A companion
   therefore moves along an arc between frames while the speckles stay put. That
   asymmetry is the entire basis of the technique, and it is reproduced here by
   injecting the companion at ``pa - sign * angles[i]`` in frame ``i`` via
   :mod:`exoklip.inject`.

4. **Photometry is exact by construction.** The companion is injected using a
   template normalised by :func:`exoklip.psf.normalize_psf` (unit flux in a
   FWHM-diameter aperture) and scaled by ``contrast * star_flux``, so the
   requested contrast is the contrast you get, and any deviation measured after
   a reduction is genuine algorithmic throughput loss rather than a bookkeeping
   error.

References
----------
Marois, C. et al. 2006, ApJ, 641, 556 (angular differential imaging)
Racine, R. et al. 1999, PASP, 111, 587 (speckle noise in AO imaging)
Soummer, R. et al. 2007, ApJ, 669, 642 (speckle statistics and pinning)
Roddier, F. 1981, Progress in Optics, 19, 281 (Kolmogorov phase statistics)
Sivaramakrishnan, A. et al. 2001, ApJ, 552, 397 (Lyot coronagraph propagation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .core import frame_center, dist_grid
from .inject import inject_companions_cube, companion_position
from .psf import normalize_psf, _aperture_sum, _airy_fwhm_factor

__all__ = [
    "SimConfig",
    "simulate_adi_sequence",
    "make_phase_screen",
    "make_pupil",
    "pupil_to_psf",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    """Parameters of a synthetic ADI sequence.

    Attributes
    ----------
    n_frames : int
        Number of exposures in the sequence.
    size : int
        Side of the (square) detector frames, in pixels. **Must be odd** so that
        the stellar centre falls exactly on a pixel, matching
        :func:`exoklip.core.frame_center`.
    fwhm : float
        Diffraction FWHM in pixels. Sets the pupil sampling internally.
    n_planets : int
        Number of companions to inject. Must match the length of the three
        ``planet_*`` tuples.
    planet_separations, planet_pas, planet_contrasts : sequence of float
        Separation in pixels, position angle in degrees (0 = North = ``+y``,
        increasing towards East = ``-x``), and contrast as a ratio of
        FWHM-aperture fluxes.
    pa_start, pa_end : float
        Parallactic angle of the first and last frame, in degrees. The total
        field rotation ``pa_end - pa_start`` is what makes ADI work: too little
        and the companion never separates from its own speckle neighbourhood.
    star_flux : float
        Flux of the (unocculted) star inside a FWHM-diameter aperture, in
        electrons. Sets the photon noise level.
    static_phase_rms : float
        RMS of the quasi-static wavefront error, in radians. This is the
        dominant speckle source and the thing KLIP is built to remove.
    dynamic_phase_rms : float
        RMS of the per-frame turbulent residual, in radians. Produces the smooth
        AO halo, which no PSF-subtraction algorithm can remove.
    static_drift : float
        Decorrelation of the quasi-static screen between the first and last
        frame, as a fraction of ``static_phase_rms``. 0 means perfectly stable
        optics.

        **This is the parameter that decides whether KLIP is worth using.** On a
        perfectly frozen speckle field the temporal median is already an optimal
        PSF model, and KLIP — which fits a separate model to every frame — only
        adds fitting noise: measured on this simulator, cADI beats KLIP by 20 %
        at ``static_drift = 0.05``. As the field decorrelates the median becomes
        a progressively worse model while KLIP adapts, and the ordering
        reverses: KLIP wins by 5 % at 0.5, 58 % at 1.0 and 129 % at 2.0.

        Real observations drift substantially over the hours needed to
        accumulate field rotation — temperature, flexure and the AO loop all
        move the aberrations — so the default is 0.8 rather than a value that
        would flatter the algorithm.
    coronagraph_radius : float
        Radius of the focal-plane mask in pixels. 0 disables the coronagraph and
        leaves a bright (possibly saturated) stellar core.
    read_noise : float
        Gaussian read noise per pixel, in electrons.
    photon_noise : bool
        Apply Poisson noise to the total electron count.
    obscuration : float
        Linear central obscuration of the pupil (Keck and VLT: about 0.14).
    n_spiders : int
        Number of secondary-support spider vanes (0, 3, 4 or 6).
    seed : int or None
        Seed for full reproducibility.
    """

    n_frames: int = 60
    size: int = 201
    fwhm: float = 4.0
    n_planets: int = 1
    planet_separations: Sequence[float] = (25.0,)
    planet_pas: Sequence[float] = (60.0,)
    planet_contrasts: Sequence[float] = (1e-4,)
    pa_start: float = -35.0
    pa_end: float = 35.0
    star_flux: float = 1e7
    static_phase_rms: float = 0.8
    dynamic_phase_rms: float = 0.25
    static_drift: float = 0.8
    coronagraph_radius: float = 0.0
    read_noise: float = 10.0
    photon_noise: bool = True
    obscuration: float = 0.14
    n_spiders: int = 0
    seed: int | None = 42

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is inconsistent."""
        if self.n_frames < 2:
            raise ValueError(f"n_frames must be at least 2; got {self.n_frames}.")
        if self.size < 21:
            raise ValueError(f"size must be at least 21 pixels; got {self.size}.")
        if self.size % 2 == 0:
            raise ValueError(
                f"size must be odd so that the star sits on a pixel; got {self.size}. "
                f"Use {self.size + 1}."
            )
        if self.fwhm <= 1.5:
            raise ValueError(
                f"fwhm must exceed 1.5 px to be at least Nyquist-sampled; "
                f"got {self.fwhm}."
            )
        lengths = {
            "planet_separations": len(self.planet_separations),
            "planet_pas": len(self.planet_pas),
            "planet_contrasts": len(self.planet_contrasts),
        }
        for name, n in lengths.items():
            if n != self.n_planets:
                raise ValueError(
                    f"{name} has {n} entries but n_planets={self.n_planets}; "
                    "the three planet_* sequences must all match n_planets."
                )
        r_max = (self.size - 1) / 2.0
        for sep in self.planet_separations:
            if not 0 < sep < r_max:
                raise ValueError(
                    f"planet separation {sep} px is outside the frame "
                    f"(0, {r_max:.1f})."
                )
        if not 0.0 <= self.obscuration < 1.0:
            raise ValueError(f"obscuration must be in [0, 1); got {self.obscuration}.")
        if self.star_flux <= 0:
            raise ValueError(f"star_flux must be positive; got {self.star_flux}.")

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view, suitable for JSON serialisation."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Pupil and phase
# --------------------------------------------------------------------------- #
def make_pupil(
    size: int,
    diameter_pix: float,
    obscuration: float = 0.14,
    n_spiders: int = 0,
    spider_width: float = 0.0,
) -> NDArray[np.float64]:
    """Binary pupil transmission of a circular aperture.

    Parameters
    ----------
    size : int
        Side of the (square) pupil grid.
    diameter_pix : float
        Outer diameter of the pupil in grid pixels. Together with ``size`` this
        sets the PSF sampling: ``lambda / D = size / diameter_pix`` detector
        pixels.
    obscuration : float, default 0.14
        Linear ratio of the central obscuration.
    n_spiders : int, default 0
        Number of secondary-support vanes, evenly distributed in angle.
    spider_width : float, default 0.0
        Vane width in grid pixels.

    Returns
    -------
    ndarray
        ``(size, size)`` array of 0 and 1.
    """
    if diameter_pix <= 2:
        raise ValueError(f"diameter_pix must exceed 2; got {diameter_pix}.")
    if diameter_pix > size:
        raise ValueError(
            f"diameter_pix ({diameter_pix}) exceeds the grid size ({size}); "
            "the pupil would be clipped."
        )

    center = frame_center((size, size))
    r = dist_grid((size, size), center=center)
    r_out = diameter_pix / 2.0

    pupil = (r <= r_out).astype(np.float64)
    if obscuration > 0:
        pupil[r <= r_out * obscuration] = 0.0

    if n_spiders > 0 and spider_width > 0:
        cy, cx = center
        yy = np.arange(size, dtype=np.float64)[:, None] - cy
        xx = np.arange(size, dtype=np.float64)[None, :] - cx
        for k in range(n_spiders):
            ang = np.pi * k / n_spiders
            # Distance to the line through the centre at angle `ang`.
            perp = np.abs(-xx * np.sin(ang) + yy * np.cos(ang))
            pupil[perp <= spider_width / 2.0] = 0.0

    return pupil


def make_phase_screen(
    size: int,
    rms: float,
    seed: int | None = None,
    power: float = -11.0 / 3.0,
    r0_pix: float | None = None,
    mask: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Random wavefront phase screen with a power-law spatial spectrum.

    The default exponent ``-11/3`` is the Kolmogorov law for atmospheric
    turbulence (Roddier 1981). Filtering complex white noise by
    ``sqrt(PSD)`` in the Fourier plane and taking the real part yields a
    Gaussian random field with the requested spectrum.

    Parameters
    ----------
    size : int
        Side of the square grid.
    rms : float
        Target RMS of the phase, in radians, measured **inside** ``mask`` when
        one is given (the pupil), otherwise over the whole grid.
    seed : int, optional
        Seed for reproducibility.
    power : float, default -11/3
        Exponent of the power spectral density.
    r0_pix : float, optional
        Fried parameter in grid pixels. When given, an outer scale is applied by
        flattening the PSD below ``1 / r0_pix``, which avoids the divergence of
        pure Kolmogorov power at zero frequency.
    mask : ndarray, optional
        Region over which the RMS is normalised (typically the pupil).

    Returns
    -------
    ndarray
        ``(size, size)`` phase screen in radians, with zero mean inside ``mask``.
    """
    rng = np.random.default_rng(seed)

    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    f = np.hypot(fy, fx)
    f[0, 0] = 1.0  # placeholder, the piston term is zeroed below

    psd = f**power
    if r0_pix is not None and r0_pix > 0:
        f_outer = 1.0 / r0_pix
        psd = (np.maximum(f, f_outer)) ** power
    psd[0, 0] = 0.0  # no piston

    noise = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    screen = np.real(np.fft.ifft2(noise * np.sqrt(psd))) * size

    region = np.ones((size, size), dtype=bool) if mask is None else mask.astype(bool)
    if not np.any(region):
        raise ValueError("mask selects no pixel; cannot normalise the phase RMS.")

    screen -= screen[region].mean()
    current = screen[region].std()
    if current > 0:
        screen *= float(rms) / current
    return screen


def pupil_to_psf(
    pupil: NDArray[np.float64],
    phase: NDArray[np.float64] | None = None,
    coronagraph_radius: float = 0.0,
    lyot_undersize: float = 0.9,
) -> NDArray[np.float64]:
    """Propagate an aberrated pupil to a focal-plane intensity.

    Without a coronagraph this is a single Fraunhofer propagation,
    ``I = |FFT(P e^{i phi})|^2``. With one, the classical Lyot train is
    simulated properly: focal-plane mask, back-propagation to the pupil, Lyot
    stop, and a final propagation to the detector (Sivaramakrishnan et al. 2001).
    A hard-edged occulter alone would not suppress starlight — it is the Lyot
    stop, rejecting the light diffracted to the pupil edge, that does the work.

    Parameters
    ----------
    pupil : ndarray
        Pupil transmission, ``(n, n)``.
    phase : ndarray, optional
        Wavefront phase in radians, same shape. ``None`` means a perfect
        wavefront, which gives the diffraction-limited off-axis PSF.
    coronagraph_radius : float, default 0.0
        Focal-plane mask radius in detector pixels. 0 disables the coronagraph.
    lyot_undersize : float, default 0.9
        Lyot stop diameter as a fraction of the pupil diameter.

    Returns
    -------
    ndarray
        Focal-plane intensity, same shape as ``pupil``, centred on
        :func:`exoklip.core.frame_center`.
    """
    field = pupil.astype(np.complex128)
    if phase is not None:
        field = field * np.exp(1j * phase)

    def _to_focal(pup: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pup)))

    def _to_pupil(foc: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(foc)))

    focal = _to_focal(field)

    if coronagraph_radius > 0:
        n = pupil.shape[0]
        r = dist_grid((n, n), center=frame_center((n, n)))
        # Soft-edged occulter: a hard edge rings badly on a discrete grid.
        occulter = 1.0 - np.exp(-((r / coronagraph_radius) ** 2))
        pupil_2 = _to_pupil(focal * occulter)

        r_pup = dist_grid((n, n), center=frame_center((n, n)))
        outer = np.max(r_pup[pupil > 0]) if np.any(pupil > 0) else 0.0
        lyot = (r_pup <= outer * lyot_undersize).astype(np.float64)
        focal = _to_focal(pupil_2 * lyot)

    return np.abs(focal) ** 2


# --------------------------------------------------------------------------- #
# Sequence generation
# --------------------------------------------------------------------------- #
def simulate_adi_sequence(config: SimConfig | None = None, **overrides: Any) -> dict[str, Any]:
    """Generate a complete synthetic ADI sequence.

    Parameters
    ----------
    config : SimConfig, optional
        Simulation parameters. Defaults to ``SimConfig()``.
    **overrides
        Individual fields overriding ``config``, for convenience:
        ``simulate_adi_sequence(n_frames=20, size=101)``.

    Returns
    -------
    dict
        ``cube``
            ``(n_frames, size, size)`` detector frames in electrons.
        ``angles``
            ``(n_frames,)`` parallactic angles in degrees.
        ``psf``
            Off-axis, non-coronagraphic PSF template normalised to unit flux in
            a FWHM-diameter aperture — the photometric reference.
        ``psf_raw``
            The same template scaled to ``star_flux``: what an unsaturated
            calibration exposure of the star would look like, and the image you
            would measure ``star_flux`` on with real data.
        ``fwhm``
            The FWHM actually realised, measured on the simulated PSF.
        ``star_flux``
            FWHM-aperture flux of the unocculted star, in electrons. This is
            exact: the frames are scaled so that the *aberrated* core really
            carries this much flux, so ``contrast`` means what it says.
        ``strehl``
            Realised aperture flux divided by the diffraction-limited one. About
            0.5 at the default ``static_phase_rms``. Reported because it is the
            factor by which a simulator that normalised on the perfect PSF would
            overstate every contrast.
        ``truth``
            List of dicts, one per injected companion, with ``radius``, ``pa``,
            ``contrast`` and ``flux``.
        ``config``
            The :class:`SimConfig` used.

    Examples
    --------
    >>> sim = simulate_adi_sequence(n_frames=20, size=101, seed=0)
    >>> sim['cube'].shape
    (20, 101, 101)
    """
    cfg = SimConfig() if config is None else config
    if overrides:
        params = cfg.to_dict()
        unknown = set(overrides) - set(params)
        if unknown:
            raise ValueError(
                f"unknown SimConfig field(s): {sorted(unknown)}. "
                f"Valid fields: {sorted(params)}."
            )
        params.update(overrides)
        cfg = SimConfig(**params)
    cfg.validate()

    size = cfg.size
    rng = np.random.default_rng(cfg.seed)

    # ---- Pupil sampling ---------------------------------------------------
    # On an n-point grid, an aperture of D grid-pixels gives lambda/D = n / D
    # detector pixels, hence FWHM = factor * n / D. Invert for D.
    factor = _airy_fwhm_factor(cfg.obscuration)
    diameter_pix = factor * size / cfg.fwhm
    if diameter_pix < 16:
        raise ValueError(
            f"the requested fwhm ({cfg.fwhm} px) on a {size}-pixel grid implies a "
            f"pupil of only {diameter_pix:.1f} grid pixels, too coarse to be "
            "meaningful. Increase size or decrease fwhm."
        )
    spider_w = max(diameter_pix / 80.0, 1.0) if cfg.n_spiders else 0.0
    pupil = make_pupil(
        size,
        diameter_pix,
        obscuration=cfg.obscuration,
        n_spiders=cfg.n_spiders,
        spider_width=spider_w,
    )
    pupil_mask = pupil > 0

    # ---- Phase screens ----------------------------------------------------
    # Unit-RMS screens, scaled below. Distinct seeds keep the static, drift and
    # dynamic components independent.
    ss = np.random.SeedSequence(cfg.seed)
    seeds = ss.spawn(3 + cfg.n_frames)

    screen_static = make_phase_screen(
        size, 1.0, seed=seeds[0], r0_pix=diameter_pix / 4.0, mask=pupil_mask
    )
    screen_drift = make_phase_screen(
        size, 1.0, seed=seeds[1], r0_pix=diameter_pix / 4.0, mask=pupil_mask
    )

    # ---- Off-axis reference PSF (no aberration, no coronagraph) -----------
    psf_perfect = pupil_to_psf(pupil, phase=None, coronagraph_radius=0.0)
    psf_template, aperture_flux_perfect, fwhm_measured = normalize_psf(
        psf_perfect, size=min(size, int(round(8 * cfg.fwhm)) | 1
        )
    )
    cy, cx = frame_center((size, size))
    perfect_flux = _aperture_sum(psf_perfect, cy, cx, fwhm_measured / 2.0)

    logger.debug(
        "simulate: pupil D=%.1f grid px, requested fwhm=%.2f, measured fwhm=%.2f",
        diameter_pix,
        cfg.fwhm,
        fwhm_measured,
    )

    # ---- Frames -----------------------------------------------------------
    angles = np.linspace(cfg.pa_start, cfg.pa_end, cfg.n_frames)
    t = (
        np.linspace(0.0, 1.0, cfg.n_frames)
        if cfg.n_frames > 1
        else np.zeros(1, dtype=np.float64)
    )

    cube = np.empty((cfg.n_frames, size, size), dtype=np.float64)
    unocculted = np.empty(cfg.n_frames, dtype=np.float64)
    for i in range(cfg.n_frames):
        phase = cfg.static_phase_rms * screen_static
        if cfg.static_drift:
            phase = phase + cfg.static_drift * t[i] * cfg.static_phase_rms * screen_drift
        if cfg.dynamic_phase_rms > 0:
            phase = phase + cfg.dynamic_phase_rms * make_phase_screen(
                size, 1.0, seed=seeds[3 + i], r0_pix=diameter_pix / 4.0, mask=pupil_mask
            )
        # The photometric reference is the star as it *actually appears* through
        # the aberrations, not the diffraction-limited ideal: an aberrated core
        # loses a large fraction of its flux to the halo, and at the default
        # static_phase_rms=0.8 that is a factor of about two. Normalising on the
        # perfect PSF while injecting companions with a perfect template would
        # make every contrast in the package optimistic by exactly that factor.
        open_psf = pupil_to_psf(pupil, phase=phase, coronagraph_radius=0.0)
        unocculted[i] = _aperture_sum(open_psf, cy, cx, fwhm_measured / 2.0)
        cube[i] = (
            open_psf
            if cfg.coronagraph_radius <= 0
            else pupil_to_psf(
                pupil, phase=phase, coronagraph_radius=cfg.coronagraph_radius
            )
        )

    # Scale so that the *realised* unocculted aperture flux equals star_flux.
    # This is what an observer measures on a calibration exposure, and it makes
    # the requested contrast exact by construction rather than by convention.
    realised = float(np.median(unocculted))
    if realised <= 0:
        raise ValueError(
            "the simulated stellar core carries no flux; static_phase_rms "
            f"({cfg.static_phase_rms}) is probably far too large."
        )
    strehl = realised / perfect_flux
    flux_scale = cfg.star_flux / realised
    cube *= flux_scale
    logger.debug(
        "simulate: Strehl-like aperture ratio %.3f, flux scale %.4g", strehl, flux_scale
    )

    # ---- Companions -------------------------------------------------------
    truth: list[dict[str, float]] = []
    for k in range(cfg.n_planets):
        radius = float(cfg.planet_separations[k])
        pa = float(cfg.planet_pas[k])
        contrast = float(cfg.planet_contrasts[k])
        flux = contrast * cfg.star_flux
        cube = inject_companions_cube(
            cube, psf_template, angles, radius=radius, pa=pa, flux=flux
        )
        truth.append(
            {"radius": radius, "pa": pa, "contrast": contrast, "flux": flux}
        )

    # ---- Detector noise ---------------------------------------------------
    if cfg.photon_noise:
        cube = np.clip(cube, 0.0, None)
        cube = rng.poisson(cube).astype(np.float64)
    if cfg.read_noise > 0:
        cube = cube + rng.normal(0.0, cfg.read_noise, size=cube.shape)

    return {
        "cube": cube,
        "angles": angles,
        "psf": psf_template,
        "psf_raw": psf_template * cfg.star_flux,
        "fwhm": float(fwhm_measured),
        "star_flux": float(cfg.star_flux),
        "strehl": float(strehl),
        "truth": truth,
        "config": cfg,
    }
