# exoklip

[![CI](https://github.com/francoisbarge8/exoklip/actions/workflows/ci.yml/badge.svg)](https://github.com/francoisbarge8/exoklip/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Post-traitement KLIP/ADI pour l'imagerie directe d'exoplanètes : soustraction de
PSF, statistiques de détection à petit échantillon, courbes de contraste
corrigées du throughput, et un simulateur en optique de Fourier permettant de
tester toute la chaîne sans télescope.

Seuls numpy et scipy sont nécessaires. astropy (FITS), matplotlib (figures) et
tqdm sont optionnels et importés paresseusement.

📖 [README in English](README.md)

![Quatre réductions de la même séquence simulée](examples/output/reductions.png)

*Une séquence simulée de 60 poses, réduite de quatre façons. Les cercles verts
marquent les deux compagnons injectés. Les échelles de couleur vont de ±25 000
pour la médiane brute à ±650 après KLIP annulaire.*

---

## Installation

```bash
git clone https://github.com/francoisbarge8/exoklip.git
cd exoklip
pip install -e ".[plot]"
```

`pip install -e .` seul donne accès à tous les algorithmes. L'extra `plot` ajoute
les figures, `fits` ajoute astropy pour lire des données réelles.

## Démarrage rapide

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

Les deux compagnons sont retrouvés dans une séquence où une simple médiane ne
montre rien. La chaîne complète, figures comprises, tourne en environ
80 secondes :

```bash
python examples/demo_full.py
```

---

## Principe

**Le problème.** Un Jupiter jeune est 10⁻⁴ à 10⁻⁶ fois moins brillant que son
étoile hôte, à une séparation de 0,1 à 1 seconde d'arc. Un coronographe supprime
le cœur stellaire, mais les erreurs de front d'onde résiduelles diffusent la
lumière en un halo de **speckles**. Chaque speckle a la forme d'une source
ponctuelle et persiste des minutes à des heures : les speckles ne se moyennent
donc pas avec le temps de pose. Ce sont eux qui limitent la détection, pas le
bruit de photons.

**L'imagerie différentielle angulaire** (Marois et al. 2006). Le télescope
observe en mode pupil-tracking, laissant le champ tourner avec le ciel au lieu de
compenser. Les speckles proviennent de l'optique et restent figés sur le
détecteur ; un compagnon est sur le ciel et décrit un arc. Cette asymétrie rend
les deux séparables : on construit un modèle de l'étoile à partir de la séquence
elle-même, on le soustrait, puis on dérote chaque résidu vers une orientation
commune avant de combiner. Le compagnon s'additionne de façon cohérente, les
speckles non.

**KLIP** (Soummer, Pueyo & Larkin 2012) choisit ce modèle de façon optimale. Pour
chaque pose, on construit une base de Karhunen-Loève à partir des vecteurs
propres de la covariance des autres poses, puis on projette sur les `K` premiers
modes. La troncature à `K` est ce qui fait fonctionner la méthode : les premiers
modes décrivent la PSF stellaire commune à toutes les poses, les suivants
commencent à ajuster le bruit propre à une pose donnée.

**L'auto-soustraction.** Le compagnon est présent dans les poses qui servent à
bâtir le modèle, donc une partie en est soustraite avec l'étoile. Augmenter le
nombre de modes améliore le modèle stellaire et retire davantage de signal
planétaire. Deux mécanismes traitent ce compromis :

- Un **seuil de rotation** (`delta_rot`) n'admet une pose dans la bibliothèque de
  références que si le compagnon s'y est déplacé d'une fraction donnée de FWHM.
  C'est ce qui protège les compagnons serrés.
- Une **correction de throughput** mesure la fraction survivante d'un compagnon
  de flux connu passé par la même réduction. Sans elle, une courbe de contraste
  est optimiste d'un facteur 2 à 10 à faible séparation. Mesuré ici : 0,41 à
  8 pixels, remontant à 0,93 à 48.

**Les statistiques de détection.** À trois éléments de résolution de l'étoile,
seuls 18 échantillons de bruit indépendants environ sont disponibles ; à 1,5,
neuf. Estimer un niveau de bruit sur neuf échantillons puis seuiller à « 5σ »
comme s'il était connu exactement surestime la significativité. Le traitement
correct est un test *t* de Student à deux échantillons (Mawet et al. 2014), dont
la pénalité croît fortement à faible séparation :

| Éléments de résolution | Rapport requis pour un vrai 5σ |
|---|---|
| 60 (au large) | **5,7** |
| 20 | **8,2** |
| 10 (≈1,6 λ/D) | **23,5** |

Annoncer le rapport brut comme un nombre de sigmas près de l'étoile est une façon
courante de publier une détection qui n'existe pas.

---

## Calibration

La courbe de contraste est vérifiée par injection plutôt que supposée correcte.
Un compagnon injecté exactement au contraste que la courbe annonce comme limite
5σ est retrouvé à **4,5 à 4,9σ**, sur deux réalisations de bruit indépendantes et
trois séparations. Les 6 % d'optimisme résiduel viennent de la dispersion
légèrement plus grande des ouvertures de référence en présence d'un compagnon.

Deux points d'implémentation que cette vérification a tranchés :

- Le simulateur normalise sur le flux d'ouverture stellaire **réalisé**, non sur
  la PSF limitée par la diffraction. Un cœur aberré perd une grande partie de son
  flux dans le halo, un facteur 2,4 à l'erreur de front d'onde par défaut, et
  normaliser sur la PSF idéale rend tous les contrastes optimistes d'exactement
  ce facteur.
- La courbe de contraste n'ajoute **pas** le biais de l'anneau résiduel au flux
  que doit porter un compagnon, car la statistique de détection est
  différentielle et le soustrait déjà. L'ajouter le compte deux fois, et comme
  KLIP sur-soustrait et que ce biais est négatif, l'erreur va dans le sens
  optimiste.

---

## KLIP comparé à l'ADI classique

KLIP ne surpasse une réduction ADI classique qu'à partir du moment où le champ de
speckles quasi-statique se décorrèle sur la séquence. Mesuré sur le simulateur :

| Dérive des speckles sur la séquence | KLIP vs ADI classique |
|---|---|
| 0,05 (optique figée) | **0,79×**, cADI gagne |
| 0,5 | 1,03× |
| 1,0 | 1,57× |
| 2,0 (forte dérive) | **2,28×** |

Sur un champ de speckles parfaitement figé, la médiane temporelle est déjà un
modèle de PSF optimal, et KLIP, qui ajuste un modèle par pose, ne fait qu'ajouter
du bruit d'ajustement. Son avantage vient de sa capacité à s'adapter à la dérive
réelle de l'optique sous l'effet de la température, des flexions et de la boucle
d'optique adaptative, pendant les heures nécessaires à accumuler de la rotation
de champ. Sur toute cette plage, le bruit résiduel de KLIP est resté à peu près
constant tandis que celui de cADI se dégradait d'un facteur trois.

Le simulateur utilise donc `static_drift = 0,8` par défaut. Une valeur plus
faible ferait paraître l'algorithme meilleur qu'il n'est.

---

## Rapport au prototype d'origine

Ce package remplace `b.py`, un script de 97 lignes qui exécutait une PCA
scikit-learn sur des patchs spatiaux recouvrants d'un seul JPEG. Cette approche
ne peut pas détecter de planète, et elle contenait par ailleurs huit défauts
d'implémentation. Chacun d'eux et son remplaçant :

| Problème | Pourquoi cela compte | Remplacé par |
|---|---|---|
| **Une PCA sur les patchs d'une seule image n'est pas du KLIP.** Une seule pose signifie ni bibliothèque de références ni rotation de champ : rien ne distingue un compagnon d'un speckle. Les deux sont compacts, brillants et localement atypiques. | Fondamental. La sortie est une carte du halo de speckles. | `klip.py` et `adi.py`, sur une séquence en rotation |
| La boucle d'itérations réajustait la PCA sur des données inchangées | Les 10 itérations étaient identiques ; la boucle n'avait aucun effet | `legacy/b_fixed.py` exclut les patchs signalés à chaque passe, le modèle converge |
| `skimage.measure.label` appliqué à un tableau RGB de flottants | Chaque valeur flottante distincte devient sa propre région : la liste de détections est du bruit | `detect.py` seuille vers un masque **binaire** avant d'étiqueter |
| Patchs recouvrants écrits avec `=` | Les derniers écrasent les premiers, ce qui jette l'essentiel du calcul | Accumulation du résidu, division par une carte de recouvrement |
| Seuil au 95ᵉ percentile de l'erreur | Signale exactement 5 % de l'image qu'une planète soit présente ou non ; ne peut jamais conclure qu'il n'y a rien | `metrics.significance_threshold`, seuil de Student-*t* à taux de faux positifs déclaré |
| `inverse_transform` appelé par patch sur ~259 000 patchs | Des minutes d'exécution pour une opération vectorisable | Entièrement vectorisé |
| Trois canaux couleur pour un détecteur infrarouge monochrome | Trois fois le coût pour des données identiques | Réduit à un seul plan |
| `resize(image, (512, 512))` | Détruit l'échantillonnage de la PSF dont dépend l'analyse | Recadrer, jamais redimensionner |
| Un JPEG 8 bits de prévisualisation de l'archive Keck | 256 niveaux ne peuvent contenir un signal à 10⁻⁵ de contraste ; la planète est quantifiée avant tout algorithme | `io.load_fits`, avec un avertissement explicite dans `io.load_image_legacy` |

`legacy/b_fixed.py` conserve l'approche mono-image avec les huit défauts
corrigés. Sa démonstration se déroule en deux parties : elle retrouve trois
pixels chauds et une trace de rayon cosmique sur une image propre, et rapporte
zéro détection lorsqu'il n'y a rien ; puis elle échoue à trouver un compagnon
300 fois plus brillant qu'une cible réaliste. La détection d'anomalies par patchs
sur une seule pose est un outil valable pour les défauts de détecteur. Ce n'est
pas de la détection de planètes.

---

## API

| Module | Contenu |
|---|---|
| `simulate` | `simulate_adi_sequence`, `SimConfig` : pupille, écran de phase de Kolmogorov, coronographe de Lyot, bruits de photon et de lecture |
| `klip` | `klip_basis`, `klip_residual`, `klip_annular`, `klip_fullframe`, `rotation_threshold_mask` |
| `adi` | `median_adi` (cADI), `pca_adi`, `klip_adi`, `optimize_n_modes` |
| `metrics` | `snr_student`, `snr_map`, `significance_threshold`, `noise_profile`, `throughput`, `contrast_curve`, `aperture_flux` |
| `detect` | `detect_sources`, `negfc_flux` (compagnon négatif), `characterize` |
| `psf` | `airy_2d`, `moffat_2d`, `gaussian_2d`, `fit_gaussian_psf`, `normalize_psf`, `fwhm_lambda_over_d` |
| `inject` | `inject_companion`, `inject_companions_cube`, `remove_companion`, `companion_position` |
| `preproc` | `bad_pixel_correction`, `find_star_center` (gaussien / Radon / symétrie), `cube_recenter`, `frame_selection`, `temporal_binning` |
| `io` | `load_fits`, `parallactic_angle`, `parallactic_angles_from_headers` (preset Keck/NIRC2), décodeur PNG natif |
| `plotting` | `plot_adi_principle`, `plot_reduction_summary`, `plot_snr_map`, `plot_contrast_curve`, `plot_throughput` |
| `rotation` | `frame_rotate`, `cube_derotate`, `frame_shift`, `cube_collapse` |
| `core` | géométrie des images, masques annulaires et sectoriels, `n_resolution_elements` |

### Conventions

- Images `(y, x)`, cubes `(n_poses, y, x)`, float64 en interne.
- Angles en **degrés** dans toute l'API publique.
- Angle de position : **0 = Nord = `+y`, croissant vers l'Est = `-x`**.
- `cube_derotate` tourne la pose *i* de `-angles[i]`, suivant VIP et pyKLIP.
- **Le contraste est un rapport de flux dans une ouverture de diamètre FWHM.**
  `normalize_psf` force le template à porter exactement 1,0 dans cette ouverture,
  ce qui donne un sens défini à chaque argument `flux`.

---

## Travailler avec des données réelles

Utilisez les **FITS** de la [Keck Observatory Archive](https://koa.ipac.caltech.edu/),
pas les JPEG de prévisualisation, qui sont en 8 bits et ne peuvent pas contenir le
signal.

```python
from exoklip.io import load_cube_from_dir, parallactic_angles_from_headers
from exoklip.preproc import bad_pixel_correction, cube_recenter
from exoklip import klip_adi, snr_map, normalize_psf

cube, headers = load_cube_from_dir("data/N2.*.fits")
angles = parallactic_angles_from_headers(headers, keys="NIRC2")
cube = bad_pixel_correction(cube)
cube, shifts = cube_recenter(cube, fwhm=4.5, method="radon")   # 'radon' résiste à la saturation
psf, star_flux, fwhm = normalize_psf(pose_de_calibration_non_saturee)
image = klip_adi(cube, angles, fwhm=fwhm, n_modes=20, delta_rot=1.0)
```

Deux sources d'erreur fréquentes. `star_flux` doit être corrigé du rapport de
temps de pose et de toute densité neutre entre l'image de calibration et les
poses scientifiques, sans quoi tous les contrastes sont décalés d'un facteur
constant. Et les angles parallactiques doivent être déroulés à travers la
discontinuité 180°/−180°, ce dont `parallactic_angles_from_headers` se charge.

Voir `examples/demo_real_data.py`.

## Interpréter les résultats

- **Travailler sur la carte de SNR, pas sur l'image réduite.** Le bruit d'une
  image réduite varie de plusieurs ordres de grandeur avec la séparation : un
  seuil unique appliqué dessus n'a pas de sens.
- **Un rapport n'est pas un nombre de sigmas.** `detect_sources` renvoie
  `threshold_5sigma` à côté de `snr` pour permettre la comparaison directe.
- **Une courbe de contraste décrit une réduction précise**, `n_modes` et
  `delta_rot` compris. La citer sans ces paramètres, ou sans correction de
  throughput, n'apporte que peu d'information.
- **Les résidus sont signés.** La sur-soustraction apparaît en lobes négatifs de
  part et d'autre d'une source, d'où la palette divergente centrée sur zéro.

## Limites

- Pas d'imagerie différentielle spectrale ni de RDI (étoile de référence).
- Pas de cartes de détection à modèle direct (ANDROMEDA, PACO, KLIP-FM). Le biais
  astrométrique et photométrique est traité par NEGFC, standard mais plus lent et
  sans budget d'erreur analytique.
- Le simulateur relève de l'optique de Fourier avec une décomposition de phase
  prescrite, non d'une simulation d'optique adaptative bout en bout. Il n'y a ni
  boucle temporelle, ni scintillation, ni effets chromatiques, ni cosmétique de
  détecteur réaliste.
- `negfc_flux` lance une réduction complète par évaluation de la fonction : il
  faut compter quelques centaines de réductions par compagnon.
- **Non validé contre une réduction publiée de données réelles.** La justesse
  repose ici sur des tests unitaires, des vérifications analytiques et des
  comparaisons avec des implémentations écrites indépendamment, ce qui constitue
  une preuve plus faible que la reproduction d'un résultat connu sur le ciel.
- Les conventions sont cohérentes entre elles, mais sans jeu de données réel
  d'orientation d'instrument connue, une inversion de signe *globale* ne serait
  pas détectable.

## Tests

```bash
pytest -q
```

81 tests, environ 45 secondes. Ce sont des tests numériques et non des smoke
tests : la base KL est vérifiée orthonormée à 1,1e-15 ; KLIP est comparé à une
implémentation SVD écrite indépendamment à 2e-15 ; les compagnons injectés
doivent être retrouvés à l'angle de position demandé pour les huit angles
cardinaux et diagonaux ; la statistique de SNR est vérifiée normale centrée
réduite sur du bruit pur par Monte-Carlo ; et `klip_annular` doit partitionner le
champ exactement, sans trou ni recouvrement entre anneaux.

## Références

- Marois, C., Lafrenière, D., Doyon, R., Macintosh, B. & Nadeau, D. 2006, *ApJ*, **641**, 556 : imagerie différentielle angulaire
- Lafrenière, D., Marois, C., Doyon, R., Nadeau, D. & Artigau, É. 2007, *ApJ*, **660**, 770 : LOCI et le critère de rotation
- Soummer, R., Pueyo, L. & Larkin, J. 2012, *ApJL*, **755**, L28 : KLIP
- Amara, A. & Quanz, S. P. 2012, *MNRAS*, **427**, 948 : PYNPOINT
- Pueyo, L. 2016, *ApJ*, **824**, 117 : modélisation directe KLIP
- Mawet, D. et al. 2014, *ApJ*, **792**, 97 : statistiques à petit échantillon
- Lagrange, A.-M. et al. 2010, *Science*, **329**, 57 : compagnon négatif
- Wertz, O. et al. 2017, *A&A*, **598**, A83 : budget d'erreur NEGFC
- Jensen-Clem, R. et al. 2018, *AJ*, **155**, 19 : conventions des courbes de contraste
- Gonzalez, C. A. G. et al. 2017, *AJ*, **154**, 7 : VIP
- Wang, J. J. et al. 2015, ascl:1506.001 : pyKLIP

Pour de la science en production, [VIP](https://github.com/vortex-exoplanet/VIP)
et [pyKLIP](https://github.com/bpiehl/pyklip) sont validés sur données réelles et
supportent bien plus d'instruments. Ce package est écrit pour être lu : chaque
formule remonte à l'article dont elle provient, et rien ne se cache derrière un
wrapper.

## Licence

MIT, voir [LICENSE](LICENSE).
