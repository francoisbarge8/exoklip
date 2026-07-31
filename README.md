# exoklip

[![CI](https://github.com/francoisb12/exoklip/actions/workflows/ci.yml/badge.svg)](https://github.com/francoisb12/exoklip/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

KLIP/ADI post-processing for direct imaging of exoplanets, with the detection
statistics and a Fourier-optics simulator to go with it.

I started this after writing a short script that ran PCA on patches of a single
telescope image and calling it KLIP. It didn't work, and figuring out *why* it
couldn't work turned into this package. The short version: with one exposure
there is nothing to separate a planet from a speckle. You need a sequence.

Only numpy and scipy are required. astropy, matplotlib and tqdm are optional.

📖 [README en français](README.fr.md)

![Four reductions of the same simulated sequence](examples/output/reductions.png)

*One simulated 60-frame sequence, reduced four ways. Green circles mark the two
injected companions. Look at the colour bars: ±25 000 in the raw median, ±650
after annular KLIP.*

---

## Install

```bash
git clone https://github.com/francoisb12/exoklip.git
cd exoklip
pip install -e ".[plot]"
```

`pip install -e .` on its own gives you every algorithm. The `plot` extra adds
the figures, `fits` adds astropy for real data.

## Quickstart

```python
from exoklip import SimConfig, simulate_adi_sequence, klip_adi, snr_map, detect_sources

sim = simulate_adi_sequence(SimConfig(n_frames=60, size=141, n_planets=2,
                                      planet_separations=(20.0, 36.0),
                                      planet_pas=(60.0, 215.0),
                                      planet_contrasts=(5e-3, 1e-3), seed=11))

image = klip_adi(sim["cube"], sim["angles"], fwhm=sim["fwhm"], n_modes=20, delta_rot=0.5)
snr = snr_map(image, sim["fwhm"], r_min=10, r_max=55)

for c in detect_sources(snr, sim["fwhm"], threshold=5.0, image=image):
    print(f"r = {c['radius']:.2f} px, PA = {c['pa']:.2f} deg, SNR = {c['snr']:.2f}")
```

```
r = 35.88 px, PA = 215.13 deg, SNR = 12.03
r = 19.64 px, PA = 59.75 deg, SNR = 8.73
```

Both companions come back out of a sequence where a plain median shows nothing.
The whole chain, with figures, takes about 80 seconds:

```bash
python examples/demo_full.py
```

---

## The physics I had to learn

**Why it's hard.** A young Jupiter is 10⁻⁴ to 10⁻⁶ times fainter than its star,
0.1 to 1 arcsecond away. A coronagraph kills the star's core, but residual
wavefront errors scatter light into a halo of **speckles**. Each speckle has
exactly the shape of a point source, and they last minutes to hours. So they
don't average out, and integrating longer doesn't help. They are the problem.

**Angular differential imaging** (Marois et al. 2006). Observe in pupil-tracking
mode: let the field rotate with the sky instead of compensating. The speckles
come from the optics, so they stay put on the detector. The companion is on the
sky, so it moves along an arc. That asymmetry is the whole trick. Build a model
of the star from the sequence itself, subtract it, rotate each residual back to a
common orientation, combine. The companion adds up. The speckles don't.

**KLIP** (Soummer, Pueyo & Larkin 2012) picks that model optimally. For each
frame it builds a Karhunen–Loève basis from the eigenvectors of the covariance of
the *other* frames, and projects onto the first `K` modes. Truncating at `K` is
the point: early modes capture the stellar PSF every frame shares, later ones
start fitting that frame's own noise.

**Self-subtraction is the catch.** The companion sits in the frames used to build
the model, so part of it gets subtracted too. More modes means a better stellar
model *and* less companion left. Two consequences, both implemented here:

- A **rotation threshold** (`delta_rot`): a frame only enters the reference
  library once the companion has moved by a set fraction of a FWHM. This is what
  protects close-in companions.
- A **throughput correction**: inject a companion of known flux, run the *same*
  reduction, see how much survives, divide. Without it a contrast curve is
  optimistic by 2 to 10× at small separation. I measure 0.41 at 8 pixels, rising
  to 0.93 at 48.

**And the statistics caught me out.** At three resolution elements from the star
you only have about 18 independent noise samples. At 1.5, nine. Estimating noise
from nine samples and then thresholding at "5σ" as if you knew it exactly
overstates your significance. The right treatment is a two-sample Student
*t*-test (Mawet et al. 2014), and the penalty blows up as you get closer:

| Resolution elements | Ratio needed for a genuine 5σ |
|---|---|
| 60 (far out) | **5.7** |
| 20 | **8.2** |
| 10 (≈1.6 λ/D) | **23.5** |

Quoting the raw ratio as "sigma" near the star is how you publish a detection
that isn't there.

**So is my contrast curve actually calibrated?** I decided to check rather than
assume. Take the contrast the curve quotes as its 5σ limit, inject a companion at
exactly that contrast, measure what comes back. Over two noise realisations and
three separations I get **4.5 to 4.9σ**, so about 6 % optimistic, which comes
from the reference apertures scattering a little more once a companion is
present.

That check found two bugs. My simulator was normalising on the diffraction-
limited PSF while the aberrated star had lost half its core flux to the halo — a
factor 2.4. And my contrast curve was adding the residual-annulus bias to the
flux a companion needs to carry, when the detection statistic already subtracts
it. That second one cost another 20 %, in the optimistic direction, because KLIP
over-subtracts and the bias is negative. Both are fixed.

---

## When is KLIP actually worth it?

This surprised me. I varied how fast the quasi-static speckle field decorrelates
over a sequence and compared against classical ADI:

| Speckle drift over the sequence | KLIP vs classical ADI |
|---|---|
| 0.05 (frozen optics) | **0.79×** — cADI wins |
| 0.5 | 1.03× |
| 1.0 | 1.57× |
| 2.0 (strong drift) | **2.28×** |

On a perfectly frozen speckle field the temporal median is already an optimal PSF
model, and KLIP just adds fitting noise on top. KLIP pays off *because* real
optics drift: temperature, flexure and the AO loop all move the aberrations over
the hours you need to build up field rotation. KLIP's residual noise stayed
roughly flat across that whole range while cADI's got three times worse.

That's why the simulator defaults to `static_drift = 0.8`. A smaller value makes
the algorithm look better than it is.

---

## What was wrong with my first attempt

The starting point was a 97-line script (`b.py`) doing scikit-learn PCA on
overlapping patches of a single JPEG. Every problem below was real.

| Problem | Why it matters | What replaces it |
|---|---|---|
| **Patch-PCA on one image is not KLIP.** One exposure means no reference library and no field rotation, so nothing tells a planet apart from a speckle. Both are compact, bright and locally unusual. | Fundamental. The output is a picture of the speckle halo, which is exactly what I got. | `klip.py` + `adi.py`, on a rotating sequence |
| The iteration loop refit PCA on unchanged data | All 10 "iterations" were identical. The loop did nothing. | `legacy/b_fixed.py` excludes flagged patches each pass so the model converges |
| `skimage.measure.label` on a float RGB array | Every distinct float value became its own region. The detection list was noise. | `detect.py` thresholds to a **binary** mask first |
| Overlapping patches written with `=` | Later patches overwrote earlier ones, so most of the computation was thrown away | Accumulate, then divide by an overlap-count map |
| Threshold at the 95th percentile | Always flags exactly 5 % of the image, planet or not. It can never say "nothing here". | `metrics.significance_threshold`, Student-*t* at a stated false-positive rate |
| `inverse_transform` per patch, ~259 000 patches | Minutes, for something vectorisable | Fully vectorised |
| Three colour channels from a monochrome IR detector | 3× the work for identical data | Collapsed to one plane |
| `resize(image, (512, 512))` | Destroys the PSF sampling the whole analysis rests on | Crop, never resize |
| An 8-bit JPEG preview from the Keck archive | 256 levels can't hold a 10⁻⁵ signal. The planet is quantised away before any algorithm runs. | `io.load_fits`, plus a loud warning in `io.load_image_legacy` |

`legacy/b_fixed.py` keeps the original single-image idea with all eight
implementation bugs fixed. Its demo makes the point in two halves: it finds three
hot pixels and a cosmic ray on a clean frame and reports zero detections when
there's nothing there, then completely fails to find a companion 300× brighter
than a real target. Patch-based anomaly detection on one frame is a fine tool for
detector defects. It is not planet detection.

---

## API

| Module | What's in it |
|---|---|
| `simulate` | `simulate_adi_sequence`, `SimConfig` — pupil, Kolmogorov phase screen, Lyot coronagraph, photon and read noise |
| `klip` | `klip_basis`, `klip_residual`, `klip_annular`, `klip_fullframe`, `rotation_threshold_mask` |
| `adi` | `median_adi` (cADI), `pca_adi`, `klip_adi`, `optimize_n_modes` |
| `metrics` | `snr_student`, `snr_map`, `significance_threshold`, `noise_profile`, `throughput`, `contrast_curve`, `aperture_flux` |
| `detect` | `detect_sources`, `negfc_flux` (negative fake companion), `characterize` |
| `psf` | `airy_2d`, `moffat_2d`, `gaussian_2d`, `fit_gaussian_psf`, `normalize_psf`, `fwhm_lambda_over_d` |
| `inject` | `inject_companion`, `inject_companions_cube`, `remove_companion`, `companion_position` |
| `preproc` | `bad_pixel_correction`, `find_star_center` (Gaussian / Radon / symmetry), `cube_recenter`, `frame_selection`, `temporal_binning` |
| `io` | `load_fits`, `parallactic_angle`, `parallactic_angles_from_headers` (Keck/NIRC2 preset), native PNG decoder |
| `plotting` | `plot_adi_principle`, `plot_reduction_summary`, `plot_snr_map`, `plot_contrast_curve`, `plot_throughput` |
| `rotation` | `frame_rotate`, `cube_derotate`, `frame_shift`, `cube_collapse` |
| `core` | frame geometry, annulus and segment masks, `n_resolution_elements` |

### Conventions

- Images are `(y, x)`, cubes are `(n_frames, y, x)`, float64 internally.
- Angles in **degrees** across the whole public API.
- Position angle: **0 = North = `+y`, increasing towards East = `-x`**.
- `cube_derotate` rotates frame *i* by `-angles[i]`, following VIP and pyKLIP.
- **Contrast means a ratio of fluxes in a FWHM-diameter aperture.**
  `normalize_psf` forces the template to carry exactly 1.0 in that aperture,
  which is what makes every `flux` argument mean something.

---

## Using real data

Get the **FITS** from the [Keck Observatory Archive](https://koa.ipac.caltech.edu/),
not the JPEG previews. Those are 8-bit and cannot hold the signal.

```python
from exoklip.io import load_cube_from_dir, parallactic_angles_from_headers
from exoklip.preproc import bad_pixel_correction, cube_recenter
from exoklip import klip_adi, snr_map, normalize_psf

cube, headers = load_cube_from_dir("data/N2.*.fits")
angles = parallactic_angles_from_headers(headers, keys="NIRC2")
cube = bad_pixel_correction(cube)
cube, shifts = cube_recenter(cube, fwhm=4.5, method="radon")   # 'radon' survives saturation
psf, star_flux, fwhm = normalize_psf(unsaturated_calibration_frame)
image = klip_adi(cube, angles, fwhm=fwhm, n_modes=20, delta_rot=1.0)
```

Two things that will bite you. `star_flux` has to be corrected for the
exposure-time ratio and any neutral-density filter between your calibration
image and your science frames, otherwise every contrast is off by a constant.
And the parallactic angles need unwrapping through the 180°/−180°
discontinuity, which `parallactic_angles_from_headers` handles.

See `examples/demo_real_data.py`.

## Reading the output

- **Use the SNR map, not the reduced image.** Noise in a reduced image varies by
  orders of magnitude with separation, so one threshold on it means nothing.
- **A ratio is not a sigma.** `detect_sources` gives you `threshold_5sigma` next
  to `snr` for that reason. Compare them.
- **A contrast curve describes one specific reduction**, `n_modes` and
  `delta_rot` included. Quoting one without those, or without a throughput
  correction, doesn't say much.
- **Residuals are signed.** Over-subtraction shows up as negative lobes beside a
  source, which is why the figures use a diverging colour map centred on zero.

## Limitations

- No spectral differential imaging, no reference-star differential imaging.
- No forward-modelled detection maps (ANDROMEDA, PACO, KLIP-FM). Astrometric and
  photometric bias goes through NEGFC instead, which is standard but slower and
  gives no analytic error budget.
- The simulator is Fourier optics with a prescribed phase-screen split, not an
  end-to-end AO simulation. No real control loop, no scintillation, no chromatic
  effects, no realistic detector cosmetics.
- `negfc_flux` runs a full reduction per function evaluation, so budget a few
  hundred reductions per companion.
- **Not validated against a published reduction of real data.** Everything here
  is backed by unit tests, analytic checks and comparison with independently
  written implementations. That is weaker than reproducing a known result on
  sky, and I want to be clear about the difference.
- Everything is self-consistent, but I have no real dataset with a known
  instrument orientation, so a *global* sign inversion would still be invisible
  to me.

## Tests

```bash
pytest -q
```

81 tests, about 45 seconds. They're numerical rather than smoke tests. The KL
basis is checked orthonormal to 1.1e-15; KLIP is compared against an
independently written SVD implementation to 2e-15; injected companions have to
come back at the requested position angle for all eight cardinal and diagonal
angles; the SNR statistic is verified standard normal on pure noise by Monte
Carlo; and `klip_annular` is checked to partition the field exactly, with no gap
and no overlap between annuli.

## References

- Marois, C., Lafrenière, D., Doyon, R., Macintosh, B. & Nadeau, D. 2006, *ApJ*, **641**, 556 — angular differential imaging
- Lafrenière, D., Marois, C., Doyon, R., Nadeau, D. & Artigau, É. 2007, *ApJ*, **660**, 770 — LOCI and the rotation criterion
- Soummer, R., Pueyo, L. & Larkin, J. 2012, *ApJL*, **755**, L28 — KLIP
- Amara, A. & Quanz, S. P. 2012, *MNRAS*, **427**, 948 — PYNPOINT
- Pueyo, L. 2016, *ApJ*, **824**, 117 — KLIP forward modelling
- Mawet, D. et al. 2014, *ApJ*, **792**, 97 — small-sample statistics
- Lagrange, A.-M. et al. 2010, *Science*, **329**, 57 — negative fake companion
- Wertz, O. et al. 2017, *A&A*, **598**, A83 — NEGFC error budget
- Jensen-Clem, R. et al. 2018, *AJ*, **155**, 19 — contrast curve conventions
- Gonzalez, C. A. G. et al. 2017, *AJ*, **154**, 7 — VIP
- Wang, J. J. et al. 2015, ascl:1506.001 — pyKLIP

If you need this for real science, use [VIP](https://github.com/vortex-exoplanet/VIP)
or [pyKLIP](https://github.com/bpiehl/pyklip). They're validated on real data and
support far more instruments. I wrote this one to be read: every formula traces
back to its paper, and nothing hides behind a wrapper.

## License

MIT — see [LICENSE](LICENSE).
