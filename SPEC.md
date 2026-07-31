> **Note historique.** Ce document est le cahier des charges rédigé *avant*
> l'implémentation, conservé parce qu'il explique les intentions de conception.
> Ce n'est pas la documentation de référence — le README l'est. Le code s'en
> écarte sur trois points, et le code a raison :
>
> - `pipeline.py` (`run_pipeline`, `PipelineConfig`) n'a jamais été écrit. Son
>   rôle est couvert par `exoklip/cli.py` et `examples/demo_full.py`.
> - Plusieurs signatures ont évolué à l'usage : `snr_student(exclude_adjacent=)`
>   plutôt que `exclude_negative=`, `throughput(injection_contrast=)` plutôt que
>   `injection_snr=`, `optimize_n_modes` sans argument `psf`, `detect_sources`
>   sans `mode=`.
> - La formule d'Airy avec obstruction porte un facteur `eps**2`, pas `eps` :
>   c'est ce qui rend l'amplitude égale à 1 au centre. Le SPEC était faux.

# exoklip — SPEC (contrat d'implémentation)

Package Python de détection d'exoplanètes par imagerie directe : KLIP / ADI,
statistiques de détection, courbes de contraste, simulateur de séquence ADI.

Remplace le prototype `Desktop/Astronomie/b.py` qui faisait une PCA sur des
patchs spatiaux d'un seul JPEG (≠ KLIP, et dominé par les speckles).

## Règles GLOBALES — non négociables

1. **Dépendances dures : `numpy` + `scipy` UNIQUEMENT.**
   - `astropy` (FITS), `matplotlib` (figures), `tqdm` (barres) sont **optionnels** :
     import paresseux (dans la fonction) + message d'erreur explicite si absent.
   - **Interdit** : `skimage`, `imageio`, `sklearn`, `numba`, `pandas`, `vip_hci`, `pyklip`.
   - Testé sur numpy 2.1.3 / scipy 1.16.2 / Python 3.12. Pas de `np.float_`,
     pas de `np.alltrue`, pas d'API numpy < 2 supprimée.
2. **Conventions d'axes** : toutes les images sont `(y, x)`, tous les cubes sont
   `(n_frames, y, x)`, `dtype=np.float64` en interne. Origine du repère au
   centre défini par `frame_center` (cf. `core.py`).
3. **Angles** : degrés partout dans les API publiques ; radians uniquement en
   interne. Angle de position (PA) d'un compagnon compté **depuis le Nord (haut,
   +y) vers l'Est (gauche, −x)**, i.e. trigonométrique + 90°, convention
   astronomique standard.
4. **Signatures exactes** : respecter à la lettre les signatures ci-dessous
   (noms des paramètres inclus — ils sont appelés par mot-clé ailleurs).
5. Chaque module : docstrings NumPy-style, avec la **référence bibliographique**
   quand la formule vient d'un papier. Type hints partout. Pas de `print` —
   utiliser `logging.getLogger(__name__)`.
6. Toute fonction publique valide ses entrées et lève `ValueError` avec un
   message actionnable (dire quelle forme a été reçue vs attendue).
7. **Aucune boucle Python sur les pixels.** Vectoriser. Boucle sur les frames
   ou sur les annuli = acceptable.
8. Pas de mutation des entrées : copier avant de modifier (`np.array(x, dtype=float, copy=True)`).
9. NaN : les cubes réels en contiennent. Utiliser `np.nanmedian`/`np.nanmean`
   là où c'est indiqué, et documenter le comportement.

---

## Couche 0 — fondations

### `exoklip/core.py`
Géométrie et masques. Aucune dépendance interne.

```python
def frame_center(shape: tuple[int, ...]) -> tuple[float, float]:
    """Centre (cy, cx) d'une frame. Convention: (n-1)/2 pour n impair,
    n/2 - 0.5 pour n pair -> donc (n-1)/2 dans les deux cas.
    Accepte shape 2D ou 3D (utilise les 2 derniers axes)."""

def dist_grid(shape, center=None) -> np.ndarray:
    """Carte 2D des distances radiales au centre, en pixels."""

def angle_grid(shape, center=None, convention: str = "trig") -> np.ndarray:
    """Carte 2D des angles en degrés dans [0, 360).
    convention='trig'  : 0 = +x, sens antihoraire.
    convention='pa'    : 0 = +y (Nord), croissant vers -x (Est)."""

def get_annulus_mask(shape, r_in: float, r_out: float, center=None) -> np.ndarray:
    """Masque booléen r_in <= r < r_out."""

def get_segment_mask(shape, r_in, r_out, pa_start, pa_end, center=None) -> np.ndarray:
    """Secteur d'anneau. pa_* en degrés, convention 'pa'. Gère le passage par 0."""

def annulus_indices(shape, r_in, r_out, center=None) -> tuple[np.ndarray, np.ndarray]:
    """(ys, xs) des pixels de l'anneau — pour l'extraction rapide de zones."""

def mask_circle(array, radius: float, center=None, fillwith=np.nan, mode="in") -> np.ndarray:
    """Masque le disque central (mode='in') ou l'extérieur (mode='out')."""

def cube_crop(cube, size: int, center=None) -> np.ndarray:
    """Recadre un cube/frame à (size, size) autour de center. size impair recommandé.
    Lève ValueError si size > dimension disponible."""

def pad_or_crop(frame, size: int) -> np.ndarray: ...

def n_resolution_elements(radius: float, fwhm: float) -> int:
    """floor(2*pi*radius / fwhm). Minimum 1. Mawet et al. 2014."""

def azimuthal_positions(radius: float, fwhm: float, center, pa_offset: float = 0.0
                        ) -> np.ndarray:
    """(n, 2) positions (y, x) des n=n_resolution_elements apertures indépendantes
    régulièrement espacées sur le cercle de rayon `radius`."""
```

### `exoklip/rotation.py`
Rotations / translations sous-pixel. Dépend de `core`.

```python
def frame_rotate(frame, angle: float, center=None, order: int = 3,
                 mode: str = "constant", cval: float = np.nan) -> np.ndarray:
    """Rotation ANTIHORAIRE de `angle` degrés autour de `center`.
    Implémentation: scipy.ndimage.affine_transform (PAS ndimage.rotate qui ne
    permet pas de choisir le centre). Les NaN d'entrée sont remplacés par 0
    avant l'interpolation puis re-masqués (sinon l'interpolation spline propage
    les NaN sur tout le voisinage). order>=2 nécessite prefilter=True."""

def cube_derotate(cube, angles, center=None, order=3, sign: float = -1.0,
                  n_jobs: int = 1) -> np.ndarray:
    """Dérotation ADI: frame i tournée de sign*angles[i] degrés.
    sign=-1 (défaut) = convention VIP/pyKLIP: on ANNULE la rotation de champ
    pour ramener le Nord en haut. Documenter explicitement."""

def frame_shift(frame, dy: float, dx: float, order=3, mode="constant",
                cval=np.nan) -> np.ndarray:
    """Translation sous-pixel (scipy.ndimage.shift). Même gestion des NaN."""

def cube_collapse(cube, mode: str = "median", weights=None, trim: float = 0.1
                  ) -> np.ndarray:
    """Combinaison temporelle: 'median'|'mean'|'trimmean'|'wmean'|'sum'.
    'trimmean' = moyenne tronquée à `trim` de chaque côté.
    'wmean' = moyenne pondérée par `weights` (ex: 1/variance annulaire).
    Utilise les variantes nan* partout."""
```

### `exoklip/psf.py`
Modèles de PSF et mesure de FWHM. Dépend de `core`.

```python
def gaussian_2d(shape, fwhm, center=None, amplitude=1.0, theta=0.0, fwhm_y=None) -> np.ndarray
def moffat_2d(shape, fwhm, beta=2.5, center=None, amplitude=1.0) -> np.ndarray
def airy_2d(shape, fwhm=None, lambda_over_d=None, center=None, amplitude=1.0,
            obscuration: float = 0.0) -> np.ndarray:
    """PSF d'Airy (pupille circulaire, obstruction centrale optionnelle).
    Relation FWHM = 1.028 * lambda/D. Utiliser scipy.special.j1 (+ j0 pour
    l'obstruction). Gérer r=0 (limite = amplitude)."""

def fwhm_lambda_over_d(wavelength_m: float, diameter_m: float, pixel_scale: float
                       ) -> float:
    """FWHM en pixels. pixel_scale en arcsec/pixel. lambda/D en rad -> arcsec
    via 206264.806. Retourne 1.028 * (lambda/D)_arcsec / pixel_scale."""

def fit_gaussian_psf(frame, center_guess=None, box: int = 15) -> dict:
    """Ajustement gaussien 2D elliptique (scipy.optimize.least_squares, loss='soft_l1').
    Retourne {'y','x','fwhm_x','fwhm_y','fwhm','theta','amplitude','offset','success'}.
    fwhm = moyenne géométrique. Robuste aux NaN (les exclut du résidu)."""

def normalize_psf(psf, fwhm=None, size=None, threshold=None) -> tuple[np.ndarray, float, float]:
    """Recentre (fit gaussien + frame_shift), recadre à `size`, et normalise
    pour que le flux dans UNE ouverture de diamètre `fwhm` vaille 1.
    Retourne (psf_normalisée, flux_aperture_initial, fwhm_mesurée).
    C'est la référence photométrique pour tous les contrastes."""

def create_synthetic_psf(size, fwhm, model="airy", obscuration=0.14, **kw) -> np.ndarray
```

---

## Couche 1 — algorithmes

### `exoklip/klip.py`
**Le cœur.** Karhunen-Loève Image Projection, Soummer, Pueyo & Larkin 2012 (ApJL 755, L28).
Dépend de `core`. Pur numpy/scipy.

Mathématique à implémenter **exactement** :
- Section (zone) extraite en vecteurs : `R` de forme `(n_ref, n_pix)`, cible `T` `(n_pix,)`.
- Chaque frame de référence ET la cible sont **centrées** : on retire la moyenne
  spatiale de la zone, frame par frame.
- Matrice de Gram `E = R @ R.T` de forme `(n_ref, n_ref)`.
  (⚠️ Gram, PAS la covariance pixel-pixel `n_pix × n_pix` : n_pix >> n_ref.)
- Diagonalisation : `scipy.linalg.eigh(E)`, valeurs propres croissantes -> inverser.
- Clipper les valeurs propres ≤ `eps * lambda_max` (défaut `eps=1e-12`) et
  réduire K en conséquence (modes numériquement nuls => division par ~0).
- Base KL : `Z = (V / sqrt(lambda)).T @ R` -> `(n_modes, n_pix)`, lignes de norme 1.
- Projection tronquée à K modes : `T_hat = (T @ Z[:K].T) @ Z[:K]`.
- Résidu : `T - T_hat`.

```python
def klip_basis(references, n_modes=None, eps: float = 1e-12,
               return_eigenvalues: bool = False):
    """Base KL orthonormée à partir de (n_ref, n_pix). Centre les références
    en interne. Retourne Z (n_modes, n_pix) [, eigenvalues]."""

def klip_project(target, basis, n_modes=None) -> np.ndarray:
    """Résidu target - projection. `target` peut être (n_pix,) ou (m, n_pix)."""

def klip_residual(target, references, n_modes, eps=1e-12) -> np.ndarray:
    """Raccourci basis+project. Doit être exactement équivalent."""

def rotation_threshold_mask(angles, index: int, radius: float, fwhm: float,
                            delta_rot: float = 1.0) -> np.ndarray:
    """Booléen (n_frames,) : True = frame utilisable comme référence pour la
    frame `index` à la séparation `radius`.
    Critère: |PA_j - PA_i| * radius * pi/180  >  delta_rot * fwhm  (déplacement
    azimutal de la planète en pixels). Marra et al. / Lafrenière et al. 2007.
    La frame `index` elle-même est TOUJOURS exclue.
    Si moins de `n_modes` références survivent, l'appelant doit relâcher le
    critère — exposer ça via `min_refs` dans klip_annular."""

def klip_annular(cube, angles, fwhm: float, n_modes=10, asize: float = 4.0,
                 delta_rot: float = 1.0, n_segments: int | str = 1,
                 r_min: float | None = None, r_max: float | None = None,
                 min_refs: int = 5, center=None, verbose: bool = False,
                 n_jobs: int = 1) -> np.ndarray:
    """KLIP annulaire (le mode de référence). Retourne le cube de RÉSIDUS
    NON dérotés (n_frames, y, x) — la dérotation est faite par `adi.py`.
    - Anneaux de largeur `asize` * fwhm pixels, de r_min (défaut fwhm) à r_max
      (défaut jusqu'au bord inscrit).
    - `n_segments`: int, ou 'auto' = max(1, floor(2*pi*r_mid / (asize*fwhm))).
    - Pour chaque frame et chaque zone : sélection des références par
      `rotation_threshold_mask` évaluée à r_mid ; si < min_refs références,
      relâcher delta_rot progressivement (×0.5 jusqu'à 3 fois) puis, en dernier
      recours, prendre les `min_refs` frames de |ΔPA| maximal, et LOGGER un warning.
    - n_modes peut être un int ou une liste -> si liste, retourne un dict
      {k: cube_residus} pour balayer K sans tout recalculer (réutiliser la base).
    - Les pixels hors zones traitées valent NaN.
    - n_jobs>1 : parallélisme sur les anneaux via concurrent.futures (stdlib)."""

def klip_fullframe(cube, angles, fwhm, n_modes=10, delta_rot=0.0,
                   mask_radius=None, center=None) -> np.ndarray:
    """PCA plein champ (delta_rot=0 => toutes les autres frames en référence).
    Plus rapide, moins performant à petite séparation. Même sortie."""
```

### `exoklip/inject.py`
Dépend de `core`, `rotation`, `psf`.

```python
def inject_companion(frame, psf_template, radius: float, pa: float, flux: float,
                     center=None, order=3) -> np.ndarray:
    """Ajoute une PSF à (radius, pa). `psf_template` DOIT être normalisée
    (flux=1 dans une ouverture FWHM) => `flux` est directement en unités de
    flux d'ouverture. Position sous-pixel via frame_shift.
    Conversion PA->cartésien: x = cx - r*sin(pa_rad), y = cy + r*cos(pa_rad)."""

def inject_companions_cube(cube, psf_template, angles, radius, pa, flux,
                           center=None, n_branches: int = 1) -> np.ndarray:
    """Injection dans une séquence ADI: la position dans la frame i doit être
    tournée de -sign*angles[i] pour que le compagnon apparaisse au bon PA
    APRÈS dérotation. `n_branches` répartit les copies à pa + k*360/n_branches.
    ⚠️ C'est LE point où on se trompe de signe — ajouter un test qui injecte,
    réduit par cADI et vérifie que le compagnon ressort au PA demandé."""

def remove_companion(cube, psf_template, angles, radius, pa, flux, center=None):
    """Injection négative (NEGFC), = inject_companions_cube avec -flux."""
```

### `exoklip/simulate.py`
Générateur de séquence ADI synthétique **réaliste** — c'est ce qui rend tout le
reste testable sans données. Dépend de `core`, `psf`, `inject`, `rotation`.

Physique à respecter :
- Pupille circulaire avec obstruction centrale + araignées optionnelles.
- Écran de phase : PSD de Kolmogorov `f^(-11/3)`, généré par FFT de bruit blanc
  complexe filtré, normalisé pour une amplitude RMS donnée (en radians).
- Décomposition **quasi-statique + turbulent** : `phase_i = phase_static +
  alpha_i * phase_dyn_i`, où `phase_static` est identique sur toute la séquence
  (=> speckles quasi-statiques corrélés temporellement, ce qui est exactement ce
  que KLIP doit soustraire et ce qu'une réduction naïve ne peut pas) et
  `phase_dyn_i` décorrélé (=> halo AO résiduel).
- Dérive lente : la phase statique évolue légèrement (`static_drift`) pour
  simuler la décorrélation sur plusieurs heures.
- PSF = |FFT(pupille * exp(i*phase))|², recadrée, normalisée.
- Coronographe : atténuation du cœur par un masque focal gaussien/Lyot simplifié
  (paramètre `coronagraph_radius`, 0 = pas de coronographe).
- Rotation de champ : les speckles restent FIXES dans le repère détecteur,
  la planète tourne avec le PA. C'est TOUT le principe de l'ADI — le test le
  plus important du package.
- Bruits : Poisson (photons) + gaussien (lecture) + flat field optionnel.

```python
@dataclass
class SimConfig:
    n_frames: int = 60
    size: int = 201
    fwhm: float = 4.0
    n_planets: int = 1
    planet_separations: tuple = (25.0,)   # pixels
    planet_pas: tuple = (60.0,)           # degrés
    planet_contrasts: tuple = (1e-4,)     # flux / flux stellaire (ouverture FWHM)
    pa_start: float = -35.0
    pa_end: float = 35.0
    star_flux: float = 1e7                # e- dans l'ouverture FWHM
    static_phase_rms: float = 0.8         # rad
    dynamic_phase_rms: float = 0.25       # rad
    static_drift: float = 0.05
    coronagraph_radius: float = 0.0
    read_noise: float = 10.0
    photon_noise: bool = True
    obscuration: float = 0.14
    seed: int | None = 42

def simulate_adi_sequence(config: SimConfig) -> dict:
    """Retourne {'cube','angles','psf','fwhm','config','truth'} où
    'psf' est la PSF hors-axe non coronographiée normalisée (référence photométrique)
    et 'truth' est la liste des (radius, pa, contrast, flux) injectés."""

def make_phase_screen(size, rms, seed=None, power=-11/3, r0_pix=None) -> np.ndarray
def make_pupil(size, diameter_pix, obscuration=0.14, n_spiders=0, spider_width=0) -> np.ndarray
def pupil_to_psf(pupil, phase, oversample=2) -> np.ndarray
```

---

## Couche 2 — réduction et statistiques

### `exoklip/adi.py`
Orchestration. Dépend de `klip`, `rotation`, `core`.

```python
def median_adi(cube, angles, center=None, collapse="median", mask_radius=None) -> np.ndarray:
    """cADI (Marois et al. 2006) : soustrait la médiane temporelle, dérote, combine.
    Baseline de comparaison obligatoire."""

def pca_adi(cube, angles, fwhm, n_modes=10, mode="annular", collapse="median",
            delta_rot=1.0, asize=4.0, n_segments=1, mask_radius=None,
            center=None, full_output=False, n_jobs=1, **kwargs):
    """Point d'entrée principal. mode='annular'|'fullframe'.
    full_output=True -> (image_finale, cube_residus_derotes, cube_residus_bruts).
    Si n_modes est une séquence -> retourne un dict {k: image} (balayage K)."""

def klip_adi(*args, **kwargs)   # alias explicite de pca_adi(mode='annular')

def optimize_n_modes(cube, angles, fwhm, psf, radius, pa, n_modes_grid,
                     **kwargs) -> dict:
    """Balaye K, calcule le SNR d'une source (vraie ou injectée) à (radius, pa)
    pour chaque K. Retourne {'n_modes': [...], 'snr': [...], 'best': K}."""
```

### `exoklip/metrics.py`
**Statistiques de détection — Mawet et al. 2014 (ApJ 792, 97).** Dépend de `core`, `inject`, `adi`.
C'est ici que le prototype `b.py` était le plus faux (seuil au 95ᵉ percentile).

```python
def aperture_flux(frame, y, x, radius, method="exact") -> float:
    """Somme dans une ouverture circulaire, avec pondération sous-pixel des
    bords ('exact' = fraction de recouvrement calculée analytiquement ou par
    sur-échantillonnage ×16 ; 'center' = appartenance binaire).
    Ignore les NaN (les compte comme 0 mais renvoie NaN si >50% de NaN)."""

def snr_student(frame, y, x, fwhm, center=None, exclude_negative=False,
                full_output=False):
    """SNR à petit échantillon, Mawet et al. 2014 eq. 9 :
        n     = nombre d'ouvertures indépendantes à cette séparation
        x_1   = flux dans l'ouverture testée
        x_bar = moyenne des n-1 autres
        s     = écart-type ÉCHANTILLON (ddof=1) des n-1 autres
        SNR   = (x_1 - x_bar) / (s * sqrt(1 + 1/(n-1)))
    Les ouvertures voisines immédiates de la cible sont exclues (contamination).
    full_output -> dict avec p-value (loi de Student à n-2 ddl, scipy.stats.t.sf),
    sigma-équivalent gaussien, n, fluxes."""

def snr_map(frame, fwhm, center=None, r_min=None, r_max=None, n_jobs=1,
            approx=False) -> np.ndarray:
    """Carte de SNR pixel à pixel. `approx=True` utilise le profil de bruit
    annulaire (rapide, biaisé à petite séparation) ; False = snr_student partout
    (lent mais correct). Paralléliser par anneaux."""

def noise_profile(frame, fwhm, center=None, method="aperture") -> dict:
    """Profil radial du bruit. method='aperture' (std des flux d'ouvertures
    indépendantes — CORRECT) ou 'pixel' (std des pixels de l'anneau — sous-estime
    le bruit d'un facteur ~sqrt(n_pix_par_ouverture), ne PAS utiliser pour un
    contraste, mais fourni pour comparaison pédagogique).
    Retourne {'radius','noise','mean'}."""

def throughput(cube, angles, psf, fwhm, radii, reduction_fn, n_branches=3,
               injection_snr=20.0, center=None, **red_kwargs) -> dict:
    """Transmission de l'algorithme (auto-soustraction + sur-soustraction).
    Pour chaque rayon et chaque branche azimutale : injecter un compagnon de
    flux connu, refaire LA MÊME réduction, mesurer le flux d'ouverture récupéré
    au bon endroit, diviser par le flux d'ouverture injecté.
    Injecter les branches SÉPARÉMENT (sinon elles se contaminent).
    Soustraire l'image de référence sans injection avant de mesurer.
    Retourne {'radius','throughput','throughput_std'}."""

def contrast_curve(cube, angles, psf, fwhm, star_flux, reduction_fn,
                   sigma=5.0, n_branches=3, radii=None, student=True,
                   center=None, **red_kwargs) -> dict:
    """Courbe de contraste 5σ corrigée du throughput et de la pénalité de
    petit échantillon.
        n(r)      = n_resolution_elements(r, fwhm)
        seuil(r)  = t.ppf(1 - FPF, n-2) * sqrt(1 + 1/(n-1))   [si student=True]
                    avec FPF = norm.sf(sigma)  (5σ -> 2.87e-7)
        contraste = (seuil * bruit(r) + biais(r)) / (throughput(r) * star_flux)
    `star_flux` = flux d'ouverture FWHM de l'étoile NON coronographiée, corrigé
    du rapport de temps de pose / de la densité neutre.
    Retourne {'radius','contrast','contrast_gaussian','throughput','noise','sigma_corr'}
    où sigma_corr est le facteur de pénalité (>1, explose sous ~2 λ/D)."""

def detection_limit_map(frame, fwhm, star_flux, throughput_dict, sigma=5.0) -> np.ndarray
```

### `exoklip/detect.py`
Détection de sources — remplace le `measure.label` sur un float RGB de `b.py`.
Dépend de `core`, `metrics`, `psf`, `inject`, `adi`.

```python
def detect_sources(snr_frame, fwhm, threshold=5.0, center=None, r_min=None,
                   r_max=None, mode="lpeaks", exclude_border=True) -> list[dict]:
    """Maxima locaux au-dessus du seuil, séparés d'au moins 1 FWHM
    (scipy.ndimage.maximum_filter + label sur le masque BINAIRE — pas sur des
    flottants). Raffinement sous-pixel par barycentre pondéré puis fit gaussien.
    Retourne une liste triée par SNR décroissant de dicts :
    {'y','x','radius','pa','snr','fwhm_fit','flux_aperture'}."""

def negfc_flux(cube, angles, psf, fwhm, radius, pa, reduction_fn,
               bounds=None, center=None, **red_kwargs) -> dict:
    """Photométrie/astrométrie par compagnon négatif (Lagrange et al. 2010,
    Wertz et al. 2017) : optimiser (r, pa, flux) pour MINIMISER la somme des
    carrés des résidus dans une ouverture de 2 FWHM autour de la position,
    après injection d'un compagnon de flux NÉGATIF et réduction complète.
    scipy.optimize.minimize (Nelder-Mead) ; initialiser sur la mesure
    d'ouverture. Retourne {'radius','pa','flux','contrast','chi2','success','n_eval'}.
    Coûteux : logger le nombre d'évaluations."""

def characterize(cube, angles, psf, fwhm, star_flux, reduction_fn,
                 threshold=5.0, do_negfc=True, **kw) -> list[dict]:
    """Pipeline détection -> caractérisation de tous les candidats."""
```

---

## Couche 3 — I/O, pipeline, restitution

### `exoklip/io.py`
```python
def load_fits(path, ext=0, return_header=False)      # astropy optionnel
def save_fits(path, data, header=None, overwrite=True)
def load_cube_from_dir(pattern, sort_by_header=None, ext=0) -> tuple[np.ndarray, list]
def parallactic_angle(ha_hours, dec_deg, lat_deg) -> float:
    """PA = arctan2( sin(HA), cos(dec)*tan(lat) - sin(dec)*cos(HA) ) en degrés.
    Vectorisé sur ha_hours."""
def parallactic_angles_from_headers(headers, keys=None, latitude=None) -> np.ndarray:
    """Cherche PARANG/PA/ROTPOSN..., sinon reconstruit depuis HA/DEC/latitude.
    Gère le déroulement (unwrap) du passage 180/-180. Presets connus:
    NIRC2/Keck (PARANG + ROTPOSN - INSTANGL, latitude 19.8283)."""
def load_image_legacy(path) -> np.ndarray:
    """Charge un JPEG/PNG SANS imageio (PNG via zlib+struct, JPEG via une
    dépendance optionnelle Pillow si présente). ÉMET UN AVERTISSEMENT FORT :
    un JPEG 8 bits ne peut pas contenir un signal à 1e-5 de contraste."""
```

### `exoklip/preproc.py`
```python
def bad_pixel_correction(cube, sigma=5.0, size=5, iterations=2, protect_mask=None)
def subtract_background(cube, method="median_annulus", r_in=None)
def find_star_center(frame, fwhm, method="gaussian", mask_radius=None, guess=None
                     ) -> tuple[float, float]:
    """'gaussian' (fit sur le cœur), 'radon' (Pueyo et al. 2015 — maximise
    l'intégrale le long de lignes passant par un centre candidat, robuste sur
    étoile saturée), 'symmetry' (corrélation avec l'image tournée de 180°)."""
def cube_recenter(cube, fwhm, method="symmetry", **kw) -> tuple[np.ndarray, np.ndarray]
def frame_selection(cube, fwhm, metric="corr", percentile=90.0) -> tuple[np.ndarray, np.ndarray]
def temporal_binning(cube, angles, n_bin) -> tuple[np.ndarray, np.ndarray]
```

### `exoklip/pipeline.py` + `exoklip/cli.py`
```python
@dataclass
class PipelineConfig:   # tous les paramètres, sérialisable en JSON
    ...
def run_pipeline(cube, angles, psf, config: PipelineConfig) -> dict:
    """preproc -> réduction (cADI + KLIP) -> SNR map -> détection -> NEGFC ->
    courbe de contraste. Retourne un dict complet + timings."""
```
CLI : `python -m exoklip demo`, `python -m exoklip run --cube ... --angles ...`,
`python -m exoklip contrast ...`. `argparse`, sortie lisible, codes de retour.

### `exoklip/plotting.py`
matplotlib **importé paresseusement**. `plot_reduction_summary`, `plot_snr_map`,
`plot_contrast_curve`, `plot_kl_modes`, `plot_adi_principle` (figure pédagogique :
speckles fixes vs planète qui tourne), `plot_throughput`. Toutes acceptent `ax=None`
et retournent la figure. Colormaps perceptuellement uniformes, échelle symétrique
centrée sur 0 pour les résidus.

---

## Tests — `tests/` (pytest)

Tests **numériques**, pas des smoke tests. Obligatoires :

- `test_core.py` : centre pair/impair, masques (comptage de pixels vs aire
  analytique à 2 % près), conventions d'angles, `n_resolution_elements`.
- `test_rotation.py` : rotation de 360° == identité (à l'interpolation près) ;
  rotation +θ puis −θ == identité ; une source à (r, pa) tournée de +90° se
  retrouve au bon endroit ; conservation du flux total à 1 % ; NaN non propagés.
- `test_psf.py` : la FWHM ajustée sur une gaussienne synthétique retrouve la
  FWHM d'entrée à 2 % ; `normalize_psf` donne bien flux_ouverture == 1 ;
  Airy : premier zéro à 1.22 λ/D.
- `test_klip.py` : **les tests critiques**
  - base orthonormée : `Z @ Z.T == I` à 1e-10 ;
  - K = n_ref-1 modes => résidu ~ 0 sur une cible qui est combinaison linéaire
    des références (à 1e-8) ;
  - le résidu est orthogonal à la base : `residual @ Z.T == 0` ;
  - reconstruction monotone : l'énergie du résidu décroît quand K croît ;
  - équivalence `klip_residual` vs `klip_basis`+`klip_project` ;
  - `rotation_threshold_mask` : vérifie la formule sur un cas calculé à la main ;
  - comparaison avec une PCA de référence (SVD directe de R) : les résidus
    doivent coïncider à 1e-8.
- `test_inject.py` : **anti-erreur de signe** — injecter à (r=25, pa=60) dans un
  cube ADI simulé, réduire par `median_adi`, retrouver le maximum à moins de
  1 pixel de la position attendue. Idem pour 3 branches.
- `test_metrics.py` : sur un bruit gaussien pur, la distribution des SNR a une
  moyenne ~0 et un écart-type ~1 ; le taux de faux positifs au seuil 5σ est
  compatible avec la théorie ; `aperture_flux` sur une image constante == aire
  de l'ouverture à 1 % ; le throughput d'une réduction identité == 1.
- `test_adi.py` : **le test bout-en-bout qui prouve que ça marche** — simuler une
  séquence avec une planète à 1e-4 de contraste noyée dans les speckles
  (SNR < 3 dans l'image brute médiane), réduire par KLIP annulaire, et exiger
  SNR > 8 à la position vraie, ET aucune détection > 8σ ailleurs.
  Vérifier aussi que KLIP bat cADI qui bat la simple médiane.
- `test_simulate.py` : les speckles sont bien statiques (corrélation entre
  frames > 0.9 après masquage de la planète), la planète bouge bien du bon
  nombre de degrés, le contraste injecté est bien celui mesuré.
- `test_regression.py` : le pipeline complet sur seed=42 donne des valeurs dans
  des bornes fixées (non-régression).

Tous les tests doivent tourner en **< 90 s au total** (utiliser des petites
images, size=101, n_frames=20-30, sauf le test bout-en-bout).

---

## Livrables annexes

- `legacy/b_fixed.py` : la version CORRIGÉE de l'approche originale (PCA sur
  patchs d'une seule image). Garder l'esprit, corriger les 8 bugs identifiés :
  itérations qui ne faisaient rien, `measure.label` sur des flottants RGB,
  écrasement des patchs qui se recouvrent, seuil au 95ᵉ percentile, boucle
  pixel par pixel, RGB inutile, redimensionnement, JPEG.
  En tête de fichier : un encadré expliquant pourquoi ce n'est PAS du KLIP et
  ce que ça peut/ne peut pas détecter. Doit tourner sans skimage/imageio.
- `examples/demo_full.py` : démo bout-en-bout sur données simulées, sauve
  `examples/output/*.png` (principe ADI, image brute vs cADI vs KLIP, carte SNR,
  courbe de contraste, modes KL, SNR vs K).
- `examples/demo_real_data.py` : squelette documenté pour des FITS NIRC2 réels.
- `README.md` : installation, quickstart, la physique en 30 lignes, tableau
  « ce qui était faux dans b.py -> ce qui le remplace », API, références.
- `requirements.txt`, `pyproject.toml` (setuptools, py>=3.10).
