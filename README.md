# exoklip

[![CI](https://github.com/francoisbarge8/exoklip/actions/workflows/ci.yml/badge.svg)](https://github.com/francoisbarge8/exoklip/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

KLIP/ADI post-processing for direct imaging of exoplanets: PSF subtraction,
small-sample detection statistics, throughput-corrected contrast curves, and a
Fourier-optics simulator for testing the whole chain without a telescope.

Only numpy and scipy are required. astropy (FITS), matplotlib (figures) and tqdm
are optional and imported lazily.

📖 [README en français](README.fr.md)

![Four reductions of the same simulated sequence](examples/output/reductions.png)

*One simulated 60-frame sequence, reduced four ways. Green circles mark the two
injected companions. The colour bars run from ±25 000 in the raw median to ±650
after annular KLIP.*

---

## Install

```bash
git clone https://github.com/francoisbarge8/exoklip.git
cd exoklip
pip install -e ".[plot]"
```

`pip install -e .` on its own gives every algorithm. The `plot` extra adds the
figures, `fits` adds astropy for reading real data.

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

Both companions are recovered from a sequence in which a plain median shows
nothing. The full chain, figures included, runs in about 80 seconds:

```bash
python examples/demo_full.py
```

---

## How it works

**The problem.** A young Jupiter is 10⁻⁴ to 10⁻⁶ times fainter than its host
star, at a separation of 0.1 to 1 arcsecond. A coronagraph suppresses the
stellar core, but residual wavefront errors scatter light into a halo of
**speckles**. Each speckle has the shape of a point source and persists for
minutes to hours, so speckles do not average down with integration time. They
are the limiting noise source, not photon noise.

**Angular differential imaging** (Marois et al. 2006). The telescope observes in
pupil-tracking mode, letting the field rotate with the sky rather than
compensating for it. Speckles originate in the optics and stay fixed on the
detector; a companion is on the sky and moves along an arc. That asymmetry makes
the two separable: a model of the star is built from the sequence itself,
subtracted, and each residual is rotated back to a common orientation before
combining. The companion adds coherently, the speckles do not.

**KLIP** (Soummer, Pueyo & Larkin 2012) selects that model optimally. For each
frame it builds a Karhunen-Loève basis from the eigenvectors of the covariance
of the other frames, then projects onto the leading `K` modes. The truncation at
`K` is what makes it work: the first modes describe the stellar PSF common to
every frame, later ones begin fitting the noise specific to a single frame.

**Self-subtraction.** The companion is present in the frames used to build the
model, so part of it is subtracted along with the star. Increasing the number of
modes improves the stellar model and removes more companion signal. Two
mechanisms address this:

- A **rotation threshold** (`delta_rot`) admits a frame into the reference
  library only once the companion has moved by a set fraction of a FWHM. This is
  what protects close-in companions.
- A **throughput correction** measures the surviving fraction of a companion of
  known flux put through the same reduction. Without it, a contrast curve is
  optimistic by a factor of 2 to 10 at small separation. Measured here: 0.41 at
  8 pixels, rising to 0.93 at 48.

**Detection statistics.** At three resolution elements from the star, only about
18 independent noise samples are available; at 1.5, nine. Estimating a noise
level from nine samples and thresholding at "5σ" as though it were known exactly
overstates significance. The correct treatment is a two-sample Student *t*-test
(Mawet et al. 2014), whose penalty grows sharply at small separation:

| Resolution elements | Ratio needed for a genuine 5σ |
|---|---|
| 60 (far out) | **5.7** |
| 20 | **8.2** |
| 10 (≈1.6 λ/D) | **23.5** |

Reporting the raw ratio as a number of sigmas close to the star is a common way
to publish a detection that is not there.

---

## Calibration

The contrast curve is verified by injection rather than assumed correct. A
companion injected at exactly the contrast the curve quotes as its 5σ limit is
recovered at **4.5 to 4.9σ**, across two independent noise realisations and
three separations. The residual 6 % of optimism comes from the reference
apertures scattering slightly more once a companion is present.

Two implementation points that this verification settled:

- The simulator normalises on the **realised** stellar aperture flux, not on the
  diffraction-limited PSF. An aberrated core loses a large fraction of its flux
  to the halo, a factor of 2.4 at the default wavefront error, and normalising
  on the ideal PSF makes every contrast optimistic by exactly that factor.
- The contrast curve does **not** add the residual-annulus bias to the flux a
  companion must carry, because the detection statistic is differential and
  already subtracts it. Adding it counts the bias twice, and since KLIP
  over-subtracts and that bias is negative, the error is in the optimistic
  direction.

---

## KLIP compared with classical ADI

KLIP outperforms a classical ADI reduction only once the quasi-static speckle
field decorrelates over the sequence. Measured on the simulator:

| Speckle drift over the sequence | KLIP vs classical ADI |
|---|---|
| 0.05 (frozen optics) | **0.79×**, cADI wins |
| 0.5 | 1.03× |
| 1.0 | 1.57× |
| 2.0 (strong drift) | **2.28×** |

On a perfectly frozen speckle field the temporal median is already an optimal
PSF model, and KLIP, which fits a separate model per frame, only adds fitting
noise. Its advantage comes from adapting as real optics drift under temperature,
flexure and the AO loop over the hours required to accumulate field rotation.
Across that whole range KLIP's residual noise stayed roughly flat while cADI's
degraded by a factor of three.

The simulator therefore defaults to `static_drift = 0.8`. A smaller value makes
the algorithm look better than it is.

---

## Relation to the original prototype

This package replaces `b.py`, a 97-line script that ran scikit-learn PCA on
overlapping spatial patches of a single JPEG. That approach cannot detect a
planet, and it also contained eight implementation defects. Each one and its
replacement:

| Problem | Why it matters | Replaced by |
|---|---|---|
| **Patch-PCA on a single image is not KLIP.** One exposure means no reference library and no field rotation, so nothing distinguishes a companion from a speckle. Both are compact, bright and locally atypical. | Fundamental. The output is a map of the speckle halo. | `klip.py` and `adi.py`, operating on a rotating sequence |
| The iteration loop refit PCA on unchanged data | All 10 iterations were identical; the loop had no effect | `legacy/b_fixed.py` excludes flagged patches on each pass so the model converges |
| `skimage.measure.label` applied to a float RGB array | Every distinct float value becomes its own region, so the detection list is noise | `detect.py` thresholds to a **binary** mask before labelling |
| Overlapping patches written with `=` | Later patches overwrite earlier ones, discarding most of the computation | Accumulate the residual, divide by an overlap-count map |
| Threshold at the 95th percentile of the error | Flags exactly 5 % of the image whether or not a planet is present; it can never report nothing found | `metrics.significance_threshold`, a Student-*t* threshold at a stated false-positive rate |
| `inverse_transform` called per patch over ~259 000 patches | Minutes of runtime for a vectorisable operation | Fully vectorised |
| Three colour channels from a monochrome IR detector | Three times the cost for identical data | Collapsed to a single plane |
| `resize(image, (512, 512))` | Destroys the PSF sampling the analysis depends on | Crop, never resize |
| An 8-bit JPEG preview from the Keck archive | 256 levels cannot hold a 10⁻⁵ contrast signal; the planet is quantised away before any algorithm runs | `io.load_fits`, with an explicit warning in `io.load_image_legacy` |

`legacy/b_fixed.py` keeps the single-image approach with all eight defects
fixed. Its demonstration runs in two parts: it recovers three hot pixels and a
cosmic ray track on a clean frame and reports zero detections when nothing is
present, then fails to find a companion 300 times brighter than a realistic
target. Patch-based anomaly detection on one frame is a valid tool for detector
defects. It is not planet detection.

---

## API

| Module | Contents |
|---|---|
| `simulate` | `simulate_adi_sequence`, `SimConfig`: pupil, Kolmogorov phase screen, Lyot coronagraph, photon and read noise |
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
- Angles are in **degrees** throughout the public API.
- Position angle: **0 = North = `+y`, increasing towards East = `-x`**.
- `cube_derotate` rotates frame *i* by `-angles[i]`, following VIP and pyKLIP.
- **Contrast is a ratio of fluxes in a FWHM-diameter aperture.**
  `normalize_psf` forces the template to carry exactly 1.0 in that aperture,
  which is what gives every `flux` argument a defined meaning.

---

## Working with real data

Use the **FITS** files from the [Keck Observatory Archive](https://koa.ipac.caltech.edu/),
not the JPEG previews, which are 8-bit and cannot hold the signal.

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

Two common sources of error. `star_flux` must be corrected for the exposure-time
ratio and any neutral-density filter between the calibration image and the
science frames, otherwise every contrast is off by a constant factor. And the
parallactic angles must be unwrapped through the 180°/−180° discontinuity, which
`parallactic_angles_from_headers` handles.

See `examples/demo_real_data.py`.

## Interpreting the output

- **Work from the SNR map, not the reduced image.** Noise in a reduced image
  varies by orders of magnitude with separation, so a single threshold applied
  to it is meaningless.
- **A ratio is not a number of sigmas.** `detect_sources` reports
  `threshold_5sigma` alongside `snr` so the two can be compared directly.
- **A contrast curve describes one specific reduction**, including its
  `n_modes` and `delta_rot`. Quoting one without those parameters, or without a
  throughput correction, carries little information.
- **Residuals are signed.** Over-subtraction appears as negative lobes flanking
  a source, which is why the figures use a diverging colour map centred on zero.

## Limitations

- No spectral differential imaging and no reference-star differential imaging.
- No forward-modelled detection maps (ANDROMEDA, PACO, KLIP-FM). Astrometric and
  photometric bias is handled by NEGFC, which is standard but slower and
  provides no analytic error budget.
- The simulator uses Fourier optics with a prescribed phase-screen decomposition,
  not an end-to-end AO simulation. There is no temporal control loop, no
  scintillation, no chromatic effects and no realistic detector cosmetics.
- `negfc_flux` runs a full reduction per function evaluation, so a few hundred
  reductions per companion should be expected.
- **Not validated against a published reduction of real data.** Correctness here
  rests on unit tests, analytic checks and comparison with independently written
  implementations, which is weaker evidence than reproducing a known on-sky
  result.
- Conventions are internally consistent, but without a real dataset of known
  instrument orientation a *global* sign inversion would not be detectable.

## Tests

```bash
pytest -q
```

81 tests, about 45 seconds. They are numerical rather than smoke tests: the KL
basis is verified orthonormal to 1.1e-15; KLIP is compared against an
independently written SVD implementation to 2e-15; injected companions must be
recovered at the requested position angle for all eight cardinal and diagonal
angles; the SNR statistic is verified standard normal on pure noise by Monte
Carlo; and `klip_annular` is checked to partition the field exactly, with no gap
and no overlap between annuli.

## References

- Marois, C., Lafrenière, D., Doyon, R., Macintosh, B. & Nadeau, D. 2006, *ApJ*, **641**, 556: angular differential imaging
- Lafrenière, D., Marois, C., Doyon, R., Nadeau, D. & Artigau, É. 2007, *ApJ*, **660**, 770: LOCI and the rotation criterion
- Soummer, R., Pueyo, L. & Larkin, J. 2012, *ApJL*, **755**, L28: KLIP
- Amara, A. & Quanz, S. P. 2012, *MNRAS*, **427**, 948: PYNPOINT
- Pueyo, L. 2016, *ApJ*, **824**, 117: KLIP forward modelling
- Mawet, D. et al. 2014, *ApJ*, **792**, 97: small-sample statistics
- Lagrange, A.-M. et al. 2010, *Science*, **329**, 57: negative fake companion
- Wertz, O. et al. 2017, *A&A*, **598**, A83: NEGFC error budget
- Jensen-Clem, R. et al. 2018, *AJ*, **155**, 19: contrast curve conventions
- Gonzalez, C. A. G. et al. 2017, *AJ*, **154**, 7: VIP
- Wang, J. J. et al. 2015, ascl:1506.001: pyKLIP

For production science, [VIP](https://github.com/vortex-exoplanet/VIP) and
[pyKLIP](https://github.com/bpiehl/pyklip) are validated on real data and support
far more instruments. This package is written to be read: every formula is
traceable to its source paper, and nothing is hidden behind a wrapper.

## License

MIT, see [LICENSE](LICENSE).
