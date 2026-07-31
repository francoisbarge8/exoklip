# exoklip

[![CI](https://github.com/francoisbarge8/exoklip/actions/workflows/ci.yml/badge.svg)](https://github.com/francoisbarge8/exoklip/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Post-traitement KLIP/ADI pour l'imagerie directe d'exoplanètes, avec les
statistiques de détection et un simulateur en optique de Fourier qui vont avec.

J'ai commencé ce projet après avoir écrit un petit script qui faisait une PCA sur
les patchs d'une seule image de télescope, en appelant ça du KLIP. Ça ne
marchait pas, et comprendre *pourquoi* ça ne pouvait pas marcher est devenu ce
package. En résumé : avec une seule pose, rien ne distingue une planète d'un
speckle. Il faut une séquence.

Seuls numpy et scipy sont nécessaires. astropy, matplotlib et tqdm sont
optionnels.

📖 [README in English](README.md)

![Quatre réductions de la même séquence simulée](examples/output/reductions.png)

*Une séquence simulée de 60 poses, réduite de quatre façons. Les cercles verts
marquent les deux compagnons injectés. Regardez les échelles de couleur : ±25 000
pour la médiane brute, ±650 après KLIP annulaire.*

---

## Installation

```bash
git clone https://github.com/francoisbarge8/exoklip.git
cd exoklip
pip install -e ".[plot]"
```

`pip install -e .` seul donne accès à tous les algorithmes. L'extra `plot` ajoute
les figures, `fits` ajoute astropy pour les données réelles.

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

Les deux compagnons ressortent d'une séquence où une simple médiane ne montre
rien. La chaîne complète, figures comprises, prend environ 80 secondes :

```bash
python examples/demo_full.py
```

---

## La physique qu'il a fallu comprendre

**Pourquoi c'est difficile.** Un Jupiter jeune est 10⁻⁴ à 10⁻⁶ fois moins
brillant que son étoile, à 0,1 à 1 seconde d'arc. Un coronographe tue le cœur
stellaire, mais les erreurs de front d'onde résiduelles diffusent la lumière en
un halo de **speckles**. Chaque speckle a exactement la forme d'une source
ponctuelle, et ils durent des minutes à des heures. Ils ne se moyennent donc pas,
et poser plus longtemps ne sert à rien. Ce sont eux, le problème.

**L'imagerie différentielle angulaire** (Marois et al. 2006). On observe en mode
*pupil-tracking* : on laisse le champ tourner avec le ciel au lieu de compenser.
Les speckles viennent de l'optique, ils restent donc figés sur le détecteur. Le
compagnon est sur le ciel, il décrit donc un arc. Toute l'astuce est là. On
construit un modèle de l'étoile à partir de la séquence elle-même, on le
soustrait, on dérote chaque résidu vers une orientation commune, on combine. Le
compagnon s'additionne. Les speckles non.

**KLIP** (Soummer, Pueyo & Larkin 2012) choisit ce modèle de façon optimale. Pour
chaque pose, on construit une base de Karhunen-Loève à partir des vecteurs
propres de la covariance des *autres* poses, et on projette sur les `K` premiers
modes. La troncature à `K` est tout l'enjeu : les premiers modes capturent la PSF
stellaire commune, les suivants commencent à ajuster le bruit propre de la pose.

**Le piège, c'est l'auto-soustraction.** Le compagnon est présent dans les poses
qui servent à bâtir le modèle, donc une partie part avec l'étoile. Plus de modes,
c'est un meilleur modèle stellaire *et* moins de compagnon restant. Deux
conséquences, toutes deux implémentées ici :

- Un **seuil de rotation** (`delta_rot`) : une pose n'entre dans la bibliothèque
  de références que si le compagnon s'est déplacé d'une fraction donnée de FWHM.
  C'est ce qui protège les compagnons serrés.
- Une **correction de throughput** : on injecte un compagnon de flux connu, on
  refait la *même* réduction, on regarde ce qui survit, on divise. Sans ça, une
  courbe de contraste est optimiste d'un facteur 2 à 10 à faible séparation. Je
  mesure 0,41 à 8 pixels, remontant à 0,93 à 48.

**Et les statistiques m'ont piégé.** À trois éléments de résolution de l'étoile,
on ne dispose que d'environ 18 échantillons de bruit indépendants. À 1,5, neuf.
Estimer un bruit sur neuf échantillons puis seuiller à « 5σ » comme si on le
connaissait exactement surestime la significativité. Le bon traitement est un
test *t* de Student à deux échantillons (Mawet et al. 2014), et la pénalité
explose quand on se rapproche :

| Éléments de résolution | Rapport requis pour un vrai 5σ |
|---|---|
| 60 (au large) | **5,7** |
| 20 | **8,2** |
| 10 (≈1,6 λ/D) | **23,5** |

Annoncer le rapport brut comme un « sigma » près de l'étoile, c'est comme ça
qu'on publie une détection qui n'existe pas.

**Ma courbe de contraste est-elle vraiment calibrée ?** J'ai préféré vérifier
plutôt que supposer. On prend le contraste que la courbe annonce comme limite 5σ,
on injecte un compagnon exactement à ce contraste, on mesure ce qui revient. Sur
deux réalisations de bruit et trois séparations, j'obtiens **4,5 à 4,9σ**, soit
environ 6 % d'optimisme, qui vient de la dispersion un peu plus grande des
ouvertures de référence en présence d'un compagnon.

Ce test a révélé deux bugs. Mon simulateur normalisait sur la PSF limitée par la
diffraction alors que l'étoile aberrée avait perdu la moitié du flux de son cœur
dans le halo : facteur 2,4. Et ma courbe ajoutait le biais de l'anneau résiduel
au flux que doit porter un compagnon, alors que la statistique de détection le
soustrait déjà. Ce second bug coûtait 20 % de plus, dans le sens optimiste,
puisque KLIP sur-soustrait et que ce biais est négatif. Les deux sont corrigés.

---

## Quand KLIP vaut-il vraiment le coup ?

Ça m'a surpris. J'ai fait varier la vitesse de décorrélation du champ de speckles
quasi-statique sur une séquence, et comparé à l'ADI classique :

| Dérive des speckles sur la séquence | KLIP vs ADI classique |
|---|---|
| 0,05 (optique figée) | **0,79×**, cADI gagne |
| 0,5 | 1,03× |
| 1,0 | 1,57× |
| 2,0 (forte dérive) | **2,28×** |

Sur un champ de speckles parfaitement figé, la médiane temporelle est déjà un
modèle de PSF optimal, et KLIP ne fait qu'ajouter du bruit d'ajustement par
dessus. KLIP paie *parce que* l'optique réelle dérive : température, flexions et
boucle d'optique adaptative déplacent les aberrations pendant les heures
nécessaires à accumuler de la rotation de champ. Le bruit résiduel de KLIP est
resté à peu près plat sur toute la plage, celui de cADI a triplé.

C'est pour ça que le simulateur utilise `static_drift = 0,8` par défaut. Une
valeur plus faible fait paraître l'algorithme meilleur qu'il n'est.

---

## Ce qui n'allait pas dans ma première tentative

Le point de départ était un script de 97 lignes (`b.py`) qui faisait une PCA
scikit-learn sur des patchs recouvrants d'un seul JPEG. Chaque problème ci-dessous
était réel.

| Problème | Pourquoi ça compte | Ce qui le remplace |
|---|---|---|
| **Une PCA sur les patchs d'une image n'est pas du KLIP.** Une seule pose, donc pas de bibliothèque de références ni de rotation de champ : rien ne distingue une planète d'un speckle. Les deux sont compacts, brillants et localement inhabituels. | Fondamental. La sortie est une image du halo de speckles, ce que j'ai effectivement obtenu. | `klip.py` + `adi.py`, sur une séquence en rotation |
| La boucle d'itérations réajustait la PCA sur des données inchangées | Les 10 « itérations » étaient identiques. La boucle ne faisait rien. | `legacy/b_fixed.py` exclut les patchs signalés à chaque passe, le modèle converge |
| `skimage.measure.label` sur un tableau RGB de flottants | Chaque valeur flottante distincte devenait sa propre région. La liste de détections était du bruit. | `detect.py` seuille d'abord vers un masque **binaire** |
| Patchs recouvrants écrits avec `=` | Les derniers écrasaient les premiers : l'essentiel du calcul partait à la poubelle | On accumule, puis on divise par une carte de recouvrement |
| Seuil au 95ᵉ percentile | Signale toujours exactement 5 % de l'image, planète ou pas. Il ne peut jamais dire « rien ici ». | `metrics.significance_threshold`, Student-*t* à taux de faux positifs déclaré |
| `inverse_transform` par patch, ~259 000 patchs | Des minutes, pour quelque chose de vectorisable | Entièrement vectorisé |
| Trois canaux couleur pour un détecteur infrarouge monochrome | 3× le travail pour des données identiques | Réduit à un seul plan |
| `resize(image, (512, 512))` | Détruit l'échantillonnage de la PSF sur lequel repose toute l'analyse | Recadrer, jamais redimensionner |
| Un JPEG 8 bits de prévisualisation de l'archive Keck | 256 niveaux ne peuvent pas contenir un signal à 10⁻⁵. La planète est quantifiée avant tout algorithme. | `io.load_fits`, plus un avertissement explicite dans `io.load_image_legacy` |

`legacy/b_fixed.py` garde l'idée d'origine (une seule image) avec les huit bugs
d'implémentation corrigés. Sa démo fait la démonstration en deux temps : elle
trouve trois pixels chauds et un rayon cosmique sur une image propre, et rapporte
zéro détection quand il n'y a rien ; puis elle échoue complètement à trouver un
compagnon 300× plus brillant qu'une cible réelle. La détection d'anomalies par
patchs sur une seule pose est un bon outil pour les défauts de détecteur. Ce
n'est pas de la détection de planètes.

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
| `preproc` | `bad_pixel_correction`, `find_star_center` (gaussien / **Radon** / symétrie), `cube_recenter`, `frame_selection`, `temporal_binning` |
| `io` | `load_fits`, `parallactic_angle`, `parallactic_angles_from_headers` (preset Keck/NIRC2), décodeur PNG natif |
| `plotting` | `plot_adi_principle`, `plot_reduction_summary`, `plot_snr_map`, `plot_contrast_curve`, `plot_throughput` |
| `rotation` | `frame_rotate`, `cube_derotate`, `frame_shift`, `cube_collapse` |
| `core` | géométrie, masques annulaires et sectoriels, `n_resolution_elements` |

### Conventions

- Images `(y, x)`, cubes `(n_poses, y, x)`, float64 en interne.
- Angles en **degrés** dans toute l'API publique.
- Angle de position : **0 = Nord = `+y`, croissant vers l'Est = `-x`**.
- `cube_derotate` tourne la pose *i* de `-angles[i]`, comme VIP et pyKLIP.
- **Le contraste est un rapport de flux dans une ouverture de diamètre FWHM.**
  `normalize_psf` force le template à porter exactement 1,0 dans cette ouverture,
  ce qui donne un sens à chaque argument `flux`.

---

## Utiliser des données réelles

Récupérez les **FITS** depuis la [Keck Observatory Archive](https://koa.ipac.caltech.edu/),
pas les JPEG de prévisualisation : ils sont en 8 bits et ne peuvent pas contenir
le signal.

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

Deux pièges classiques. `star_flux` doit être corrigé du rapport de temps de pose
et de toute densité neutre entre l'image de calibration et les poses
scientifiques, sinon tous les contrastes sont décalés d'un facteur constant. Et
les angles parallactiques doivent être déroulés à travers la discontinuité
180°/−180°, ce dont `parallactic_angles_from_headers` se charge.

Voir `examples/demo_real_data.py`.

## Lire les résultats

- **La carte de SNR, pas l'image réduite.** Le bruit d'une image réduite varie de
  plusieurs ordres de grandeur avec la séparation : un seuil unique dessus ne
  veut rien dire.
- **Un rapport n'est pas un sigma.** `detect_sources` renvoie `threshold_5sigma`
  à côté de `snr` pour cette raison. Comparez-les.
- **Une courbe de contraste décrit une réduction précise**, `n_modes` et
  `delta_rot` compris. La citer sans ces valeurs, ou sans correction de
  throughput, ne dit pas grand-chose.
- **Les résidus sont signés.** La sur-soustraction apparaît en lobes négatifs de
  part et d'autre d'une source ; d'où la palette divergente centrée sur zéro.

## Limites

- Pas d'imagerie différentielle spectrale, pas de RDI (étoile de référence).
- Pas de cartes de détection à modèle direct (ANDROMEDA, PACO, KLIP-FM). Le biais
  astrométrique et photométrique passe par NEGFC, standard mais plus lent et sans
  budget d'erreur analytique.
- Le simulateur relève de l'optique de Fourier avec une décomposition de phase
  prescrite, pas d'une simulation d'optique adaptative bout en bout. Ni boucle
  temporelle réelle, ni scintillation, ni effets chromatiques, ni cosmétique de
  détecteur réaliste.
- `negfc_flux` lance une réduction complète par évaluation de la fonction :
  comptez quelques centaines de réductions par compagnon.
- **Non validé contre une réduction publiée de données réelles.** Tout ce qui est
  ici repose sur des tests unitaires, des vérifications analytiques et des
  comparaisons avec des implémentations écrites indépendamment. C'est plus faible
  que reproduire un résultat connu sur le ciel, et je tiens à le dire clairement.
- Tout est cohérent en interne, mais je n'ai pas de jeu de données réel avec une
  orientation d'instrument connue : une inversion de signe *globale* me resterait
  invisible.

## Tests

```bash
pytest -q
```

81 tests, environ 45 secondes. Ce sont des tests numériques, pas des smoke tests.
La base KL est vérifiée orthonormée à 1,1e-15 ; KLIP est comparé à une
implémentation SVD écrite indépendamment à 2e-15 ; les compagnons injectés
doivent revenir à l'angle de position demandé pour les huit angles cardinaux et
diagonaux ; la statistique de SNR est vérifiée normale centrée réduite sur du
bruit pur par Monte-Carlo ; et `klip_annular` doit partitionner le champ
exactement, sans trou ni recouvrement entre anneaux.

## Références

- Marois, C., Lafrenière, D., Doyon, R., Macintosh, B. & Nadeau, D. 2006, *ApJ*, **641**, 556: imagerie différentielle angulaire
- Lafrenière, D., Marois, C., Doyon, R., Nadeau, D. & Artigau, É. 2007, *ApJ*, **660**, 770: LOCI et le critère de rotation
- Soummer, R., Pueyo, L. & Larkin, J. 2012, *ApJL*, **755**, L28: KLIP
- Amara, A. & Quanz, S. P. 2012, *MNRAS*, **427**, 948: PYNPOINT
- Pueyo, L. 2016, *ApJ*, **824**, 117: modélisation directe KLIP
- Mawet, D. et al. 2014, *ApJ*, **792**, 97: statistiques à petit échantillon
- Lagrange, A.-M. et al. 2010, *Science*, **329**, 57: compagnon négatif
- Wertz, O. et al. 2017, *A&A*, **598**, A83: budget d'erreur NEGFC
- Jensen-Clem, R. et al. 2018, *AJ*, **155**, 19: conventions des courbes de contraste
- Gonzalez, C. A. G. et al. 2017, *AJ*, **154**, 7: VIP
- Wang, J. J. et al. 2015, ascl:1506.001: pyKLIP

Pour de la science en production, utilisez [VIP](https://github.com/vortex-exoplanet/VIP)
ou [pyKLIP](https://github.com/bpiehl/pyklip) : ils sont validés sur données
réelles et supportent bien plus d'instruments. J'ai écrit celui-ci pour qu'il se
lise : chaque formule remonte à son article, et rien ne se cache derrière un
wrapper.

## Licence

MIT, voir [LICENSE](LICENSE).
