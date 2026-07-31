# exoklip

[![CI](https://github.com/francoisb12/exoklip/actions/workflows/ci.yml/badge.svg)](https://github.com/francoisb12/exoklip/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Direct imaging of exoplanets: **KLIP/ADI post-processing, honest detection
statistics, and a Fourier-optics simulator** to test it all without a telescope.

Finding a planet next to its star means recovering a source ten thousand to a
million times fainter, a fraction of an arcsecond away. The planet is not hidden
by darkness but by *speckles* — diffracted starlight that mimics point sources
and does not average away. This package implements the standard machinery for
removing them and, just as importantly, for stating truthfully what was and was
not detected afterwards.

Hard dependencies are **numpy and scipy only**. astropy, matplotlib and tqdm are
optional and imported lazily.

📖 [Version française du README](README.fr.md)

![Four reductions of the same simulated sequence](examples/output/reductions.png)

*The same 60-frame simulated sequence, reduced four ways. Green circles mark the
two injected companions. Note the colour bars: from ±25 000 in the raw median to
±650 after annular KLIP — two orders of magnitude of starlight removed.*

---

## Install

```bash
git clone https://github.com/francoisb12/exoklip.git
cd exoklip
pip install -e ".[plot]"
```

`pip install -e .` alone gives you the full algorithms; the `plot` extra only
adds the figures, and `fits` adds astropy for reading real data.

## Quickstart

```python
from exoklip import SimConfig, simulate_adi_sequence, klip_adi, snr_map, detect_sources

sim = simulate_adi_sequence(SimConfig(n_frames=60, size=141, n_planets=2,
                                      planet_separations=(20.0, 36.0),
                                      planet_pas=(60.0, 215.0),
                                      planet_contrasts=(2e-3, 5e-4), seed=11))

image = klip_adi(sim["cube"], sim["angles"], fwhm=sim["fwhm"], n_modes=20, delta_rot=0.5)
snr = snr_map(image, sim["fwhm"], r_min=10, r_max=55)

for c in detect_sources(snr, sim["fwhm"], threshold=5.0, image=image):
    print(f"r = {c['radius']:.2f} px, PA = {c['pa']:.2f} deg, SNR = {c['snr']:.2f}")
```

```
r = 35.94 px, PA = 215.01 deg, SNR = 16.16
r = 19.63 px, PA = 60.53 deg, SNR = 8.66
```

Both companions recovered from a sequence where a plain median shows nothing.
The full chain — simulation, three reductions, detection, throughput, contrast
curve, five figures — runs in about 45 seconds:

```bash
python examples/demo_full.py
```

---

## The physics, in about thirty lines

**The problem.** A young Jupiter is 10⁻⁴ to 10⁻⁶ times as bright as its star and
sits 0.1–1 arcsecond away. A coronagraph suppresses the star's core, but the
residual wavefront errors of the telescope scatter light into a halo of
**speckles** — coherent diffraction artefacts, each shaped exactly like a point
source. They are not random noise: they persist for minutes to hours, so
integrating longer does not remove them. They are the limiting factor.

**The trick: angular differential imaging** (Marois et al. 2006). Observe in
*pupil-tracking* mode: let the telescope's field rotate with the sky instead of
compensating for it. The speckles come from the optics, so they stay pinned to
the detector. The companion is on the sky, so it moves along an arc. Now the two
are separable — build a model of the star from the sequence itself, subtract it,
rotate each residual back to a common sky orientation, and combine. The
companion adds up; the speckles do not.

**KLIP** (Soummer, Pueyo & Larkin 2012) chooses that model optimally. For each
frame, it builds a Karhunen–Loève basis from the eigenvectors of the covariance
of the *other* frames and projects onto the leading `K` modes. Truncating at `K`
is the whole point: the first modes capture the stellar PSF that every frame
shares, the later ones start describing that frame's individual noise.

**The catch: self-subtraction.** The companion is present in the frames used to
build the model, so part of it is subtracted along with the star. More modes
means a cleaner stellar model *and* less companion left. Two things follow, and
both are implemented here:

- A **rotation threshold** (`delta_rot`): a frame is only allowed into the
  reference library once the companion has moved by a set fraction of a FWHM
  between the two exposures. This is what protects close-in companions.
- A **throughput correction**: measure what fraction of a known injected
  companion survives the *same* reduction, and divide. A contrast curve without
  it is optimistic by a factor of 2–10 at small separation. Measured here: 0.43
  at 8 pixels rising to 0.88 at 48.

**And the statistics.** At three resolution elements from the star there are
only about 18 independent noise samples available; at 1.5, only 9. Estimating a
noise level from 9 samples and thresholding at "5σ" as though it were known
exactly over-reports significance. The correct treatment is a two-sample Student
*t*-test (Mawet et al. 2014), which costs a penalty that diverges as the
separation shrinks:

| Resolution elements | Ratio needed for genuine 5σ |
|---|---|
| 60 (far out) | **5.7** |
| 20 | **8.2** |
| 10 (≈1.6 λ/D) | **23.5** |

Reporting the raw ratio as "sigma" close to the star is the most common way to
publish a detection that is not there.

---

## When is KLIP actually worth it?

A finding from building this, measured on the simulator by varying how fast the
quasi-static speckle field decorrelates over the sequence:

| Speckle drift over the sequence | KLIP vs classical ADI |
|---|---|
| 0.05 (frozen optics) | **0.82×** — cADI wins |
| 0.5 | 1.05× |
| 1.0 | 1.58× |
| 2.0 (strong drift) | **2.29×** |

On a perfectly frozen speckle field the temporal median is already an optimal
PSF model, and KLIP — which fits a separate model per frame — only adds fitting
noise. KLIP pays off precisely because real optics drift: temperature, flexure
and the AO loop all move the aberrations over the hours needed to accumulate
field rotation. Its residual noise stayed near-constant across that whole range
while cADI's degraded by a factor of three.

This is why the simulator's default `static_drift` is 0.8 and not a value that
would flatter the algorithm.

---

## What was wrong with the original prototype

This package grew out of a 97-line script (`b.py`) that ran scikit-learn PCA on
overlapping spatial patches of a single JPEG and called it KLIP. Every defect
below was real, and each row names what replaces it.

| Problem | Consequence for the science | Replaced by |
|---|---|---|
| **Patch-PCA on a single image is not KLIP.** With one exposure there is no reference library and no field rotation, so nothing distinguishes a planet from a speckle — both are compact, bright and locally atypical. | Fundamental: the method cannot do what it claims, and its output is a map of the speckle halo. | `klip.py` + `adi.py`: a real reference library across a rotating sequence |
| The iteration loop refits PCA on unchanged data | All 10 "iterations" identical; the loop was decorative | `legacy/b_fixed.py` — iteratively excludes flagged patches so the model converges |
| `skimage.measure.label` applied to a float RGB array | Labels every distinct float value as its own region; the detection list is noise | `detect.py` thresholds to a **binary** mask first, then labels |
| Overlapping patches written with `=` | Later patches overwrite earlier ones; most of the computation is discarded | Accumulate residual and divide by an overlap-count map |
| Threshold at the 95th percentile of the error | Always flags exactly 5 % of the image, planet or no planet — it cannot return "nothing found" | `metrics.significance_threshold`: Student-*t* at a stated false-positive rate |
| Per-patch `inverse_transform` in a Python loop over ~259 000 patches | Minutes of runtime for a vectorisable operation | Fully vectorised |
| Three colour channels from a monochrome IR detector | 3× the cost for redundant data | Collapsed to one plane |
| `resize(image, (512, 512))` | Destroys the PSF sampling that the whole analysis depends on | Crop, never resize |
| Reading an 8-bit JPEG preview from the Keck archive | 256 levels cannot hold a 10⁻⁵ contrast signal; the planet is quantised away before any algorithm runs | `io.load_fits` + a loud warning in `io.load_image_legacy` |

`legacy/b_fixed.py` keeps the original single-image idea and fixes all eight
implementation bugs, with a banner explaining what such a method can and cannot
legitimately do. Patch-based anomaly detection on one frame is a reasonable tool
— for finding cosmic rays, detector defects or extended structure. It is not
planet detection.

---

## API tour

| Module | What it gives you |
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
- Angles are in **degrees** everywhere in the public API.
- Position angle: **0 = North = `+y`, increasing towards East = `-x`.**
- `cube_derotate` rotates frame *i* by `-angles[i]` (the VIP/pyKLIP convention).
- **Contrast is a ratio of fluxes in a FWHM-diameter aperture.** `normalize_psf`
  enforces that the template carries exactly 1.0 in such an aperture, which is
  what makes every `flux` argument directly interpretable.

---

## Using real data

Get the **FITS** files from the [Keck Observatory Archive](https://koa.ipac.caltech.edu/),
not the JPEG previews — those are 8-bit and cannot hold the signal.

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

Two things that will bite you: `star_flux` must be corrected for the
exposure-time ratio and any neutral-density filter between the calibration image
and the science frames, or every contrast is off by a constant factor; and the
parallactic angles must be *unwrapped* through the 180°/−180° discontinuity,
which `parallactic_angles_from_headers` does.

See `examples/demo_real_data.py`.

## Reading the output

- **SNR map, not the reduced image.** Noise in a reduced image varies by orders
  of magnitude with separation, so a single threshold on it is meaningless.
- **A ratio is not a sigma.** `detect_sources` reports `threshold_5sigma`
  alongside `snr` for exactly this reason. Compare them.
- **A contrast curve is a statement about a specific reduction**, including its
  `n_modes` and `delta_rot`. Quoting one without those numbers, or without a
  throughput correction, says very little.
- **Residuals are signed.** Over-subtraction shows up as negative lobes flanking
  a source; that is why the figures use a diverging colour map centred on zero.

## Limitations

Honestly stated:

- No spectral differential imaging, no reference-star differential imaging.
- No forward-modelled detection maps (ANDROMEDA, PACO, KLIP-FM). Astrometric and
  photometric bias is handled by NEGFC, which is standard but slower and does not
  give an analytic error budget.
- The simulator is Fourier optics with a prescribed phase-screen split, not an
  end-to-end AO simulation: no real temporal control loop, no scintillation, no
  chromatic effects, no realistic detector cosmetics.
- `negfc_flux` runs a full reduction per function evaluation, so expect a few
  hundred reductions per companion.
- Not validated against a published reduction of a real dataset. Every
  correctness claim here comes from unit tests, analytic checks and comparison
  with independent implementations — that is weaker evidence than reproducing a
  known result on sky.

## Tests

```bash
pytest -q
```

74 tests, about 25 seconds. They are numerical, not smoke tests: the KL basis is
checked to be orthonormal to 1.1e-15, KLIP is compared against an independently
written SVD implementation to 2e-15, injected companions must return at the
requested position angle for all eight cardinal and diagonal angles, and the
SNR statistic is verified to be standard normal on pure noise by Monte Carlo.

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

For production science, use [VIP](https://github.com/vortex-exoplanet/VIP) or
[pyKLIP](https://github.com/bpiehl/pyklip): they are validated on real data and
have far more instrument support. This package is built to be *read* — every
formula is traceable to its paper, and nothing is hidden behind a wrapper.

## License

MIT — see [LICENSE](LICENSE).
