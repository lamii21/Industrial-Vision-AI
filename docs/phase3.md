# Phase 3 : modèles préentraînés et comparaison expérimentale

> Le protocole est figé dans [phase3_protocol.md](phase3_protocol.md) (commit `cdc70ce`).
> Ce document décrit **l'implémentation**, puis, après l'étape 4, le déroulé et les résultats.
> Toutes les mesures portent sur la **validation**. Le **test reste verrouillé**.

| Étape | Contenu | Statut |
|---|---|---|
| 1 | Préparation technique : versions, architectures, déterminisme, budget | validée |
| 2a | Protocole gelé (`docs/phase3_protocol.md`) | validée, commit `cdc70ce` |
| 2b | Implémentation, tests, documentation | **ce document** |
| 3 | Validation avant entraînement : tests, configurations, relance de non-régression | à faire |
| 4 | Expériences E1 à E5 | à faire (après validation) |

## 1. Architecture des nouveaux modules

```
ai-engine/
├── ivai/
│   ├── models/
│   │   ├── pretrained.py      ResNet50 / MobileNetV2 torchvision : tête à 6 sorties, poids ImageNet V1
│   │   │                      lus dans le cache local uniquement, sonde linéaire (réseau gelé, BN en eval)
│   │   └── registry.py        build_model(cfg), default_input(cfg), parameter_groups(...)
│   ├── data/
│   │   ├── dataset.py         (modifié) normalisation "dataset" | "imagenet", réplication du canal gris
│   │   └── subsets.py         sous-ensembles stratifiés emboîtés (E3), époques virtuelles de 1 260 échantillons
│   ├── training/
│   │   └── trainer.py         (modifié) groupes de paramètres (lr du réseau et lr de la tête)
│   └── evaluation/
│       ├── calibration.py     NLL, Brier, ECE (largeur et effectifs égaux), bootstrap, bruit de référence
│       ├── stats.py           Clopper-Pearson, McNemar exact, règle NLL appariée, sélection figée (§10)
│       ├── comparison.py      chargement des runs, synthèse sur les graines, courbes d'apprentissage
│       ├── benchmark.py       latence (médiane, IQR), MAC, taille, alimentation secteur
│       └── robustness.py      perturbations déterministes de la validation (E5)
├── configs/                   resnet50_ft, mobilenetv2_ft, mobilenetv2_scratch, resnet50_probe, mobilenetv2_probe
├── scripts/
│   ├── train.py               (modifié) tous les modèles, --train-fraction, aucun écrasement, aucun téléchargement
│   ├── download_weights.py    SEUL point de téléchargement des poids ImageNet (exécuté le 16/11/2026, voir §6 bis)
│   ├── benchmark_models.py    E4
│   ├── evaluate_robustness.py E5
│   └── compare_models.py      tableaux, comparaisons appariées, sélection, figures
└── tests/                     test_pretrained, test_subsets, test_phase3_pipeline, test_calibration_stats,
                               test_benchmark_robustness, test_comparison
```

**Flux d'un run :**
1. configuration YAML ;
2. `registry.build_model` et `default_input` : 1 canal avec les statistiques du train, ou 3 canaux avec ImageNet ;
3. `dataset.build_transforms` ;
4. `subsets` (E3 uniquement) ;
5. `registry.parameter_groups` ;
6. `trainer.fit` ;
7. `run.json`, `history.csv`, `val_predictions.csv` et, pour E3, `train_subset.csv`.

La comparaison relit uniquement ces fichiers, c'est-à-dire la validation.

## 2. Choix d'implémentation

### 2.1 Modèles torchvision (`pretrained.py`)
- **Ordre de construction identique avec ou sans préentraînement** :
  1. `resnet50(weights=None)`, qui initialise aléatoirement tous les poids ;
  2. chargement des poids ImageNet (si `pretrained`) ;
  3. création de la nouvelle tête.

  Le générateur aléatoire est donc consommé de la même façon dans les deux cas : MobileNetV2 préentraîné et MobileNetV2 depuis zéro ne diffèrent **que** par les poids chargés.
- **Aucun téléchargement implicite.**
  - Les poids sont lus dans `~/.cache/torch/hub/checkpoints`. S'ils manquent, `WeightsNotAvailableError` est levée.
  - `train.py` vérifie leur présence **avant** de créer le moindre dossier.
  - Seul `scripts/download_weights.py` télécharge (`allow_download=True`), en vérifiant le hash.
- **Sonde linéaire.**
  - Tous les paramètres hors tête passent à `requires_grad=False`.
  - Le modèle devient une **sous-classe** dont `train()` remet le réseau en mode eval : les statistiques BatchNorm restent figées et le Dropout du réseau est désactivé.
  - Une sous-classe plutôt qu'une méthode patchée sur l'instance : `copy.deepcopy` conserve ainsi le comportement (vérifié par un test).
- **`AdaptiveAvgPool2d` conservé** : avec une sortie 1×1, il s'exécute comme `mean`, qui est déterministe (protocole §13.2). Un test vérifie l'opération réellement exécutée.

### 2.2 Données (`dataset.py`, `subsets.py`)
- `build_transforms(train, mean, std, aug, channels=1)`. Avec les arguments de la phase 2, la liste d'opérations et leurs paramètres sont **inchangés** (vérifié par test).
- Pour `channels=3`, l'image grise est augmentée (flips, luminosité, contraste) **puis** répliquée sur 3 canaux, juste avant la normalisation ImageNet. Les 3 canaux reçoivent donc exactement la même augmentation.
- Sous-ensembles d'E3 :
  - générateur `numpy.random.default_rng(graine)` ; le générateur global de torch n'est pas touché (testé) ;
  - fichiers triés par nom, permutation par classe, k premiers ;
  - indépendants de l'ordre des lignes du CSV (testé).
- Époques virtuelles : `RandomSampler(num_samples=1260, replacement=False)`. À la fraction 1, `train.py` conserve la construction historique `shuffle=True`. L'équivalence des deux constructions est tout de même testée.

### 2.3 Trainer et `train.py`
- `fit(..., param_groups=None)` : **sans groupes, c'est exactement le chemin de la phase 2**, soit `Adam(model.parameters())`.
- Avec groupes, `ReduceLROnPlateau` divise tous les groupes par le même facteur. Le rapport 10 entre le taux de la tête et celui du réseau est donc conservé (testé). `history.csv` ajoute alors les colonnes `lr_backbone` et `lr_head` ; avec un seul groupe, les colonnes historiques sont inchangées (testé).
- Dans `train.py`, pour le CNN personnalisé, les appels qui consomment le générateur aléatoire sont faits **dans le même ordre** qu'en phase 2 :
  - `set_seed` ;
  - jeux de données ;
  - loader ;
  - modèle ;
  - `count_parameters` ;
  - champ réceptif théorique, puis mesuré ;
  - `fit`.
- **Aucun écrasement** : arrêt si le dossier du run ou le checkpoint existe déjà, ce qui protège aussi les dossiers des phases 1 et 2.
- `run.json` enregistre en plus :
  - les poids préentraînés (nom et URL) ;
  - l'entrée (canaux, normalisation) ;
  - le sous-ensemble (fraction annoncée et réelle, effectifs, règle de tirage) ;
  - les groupes de l'optimiseur (nom, taux, nombre de paramètres) ;
  - le nombre de pas par époque.

### 2.4 Comparaison et sélection
- `stats.select_default_model` applique **littéralement** le §10 du protocole, dont les constantes sont reprises telles quelles :
  - marge de 2 images ;
  - rappel ≥ 0,95 ;
  - 10 000 tirages bootstrap avec la graine 0 ;
  - égalité de latence en dessous de 5 %.

  Les candidats sont les trois modèles d'E1. Sans benchmark E4, ou avec un benchmark de type `--quick`, la sélection reste « pending ».
- `compare_models.py` vérifie que les modèles comparés ont **les mêmes graines et les mêmes images de validation dans le même ordre** (`check_aligned`). Il refuse tout run dont `run.json` indique `test_used: true`.
- ECE : intervalle bootstrap, bruit de référence, variante à effectifs égaux et occupation des intervalles. Elle reste **descriptive**.
- Diagramme de fiabilité : il affiche l'effectif de chaque intervalle.

### 2.5 Benchmark (E4) et robustesse (E5)
- La latence est mesurée sur le checkpoint de la graine 42 : elle ne dépend pas de la valeur des poids, et le checkpoint garantit l'architecture exacte.
- Les modèles sont reconstruits **sans poids ImageNet** : le checkpoint contient tous les paramètres.
- Le benchmark refuse de tourner sur batterie (`GetSystemPowerStatus`), sauf avec `--allow-battery`, réservé aux essais à blanc.
- JPEG d'E5 réalisé avec **Pillow**, car l'encodeur JPEG de torchvision est déprécié depuis la version 0.29. Bruit avec la graine 0 + l'indice de l'image.

## 3. Tests

**137 tests, tous passent, aucun ignoré** depuis le téléchargement des poids (`python -m pytest`, environ 4 min, GPU disponible). Avant ce téléchargement, les 2 tests qui comparent les poids chargés aux poids ImageNet de référence étaient ignorés.

Le fichier ResNet50 V1 (2017) ne contient pas les 53 buffers BatchNorm `num_batches_tracked`, apparus avec PyTorch 0.4.1. PyTorch les remet à 0 au chargement strict. Ils n'ont aucun effet, car toutes les BatchNorm ont `momentum=0.1` et ces compteurs ne servent que si `momentum=None`. Le test vérifie que les seules clés absentes sont ces buffers et que tous les autres tenseurs sont identiques à la référence.

| Fichier | Tests | Vérifie |
|---|---|---|
| tests des phases 1 et 2 (7 fichiers) | 49 | **inchangés, tous verts** |
| `test_pretrained.py` | 25 (dont la plupart paramétrés sur les 2 architectures) | 23 520 326 et 2 231 558 paramètres ; têtes de 12 294 et 7 686 paramètres ; seule la dernière couche diffère de torchvision ; 6 logits ; GAP exécuté comme `mean` ; **aucun téléchargement implicite** ; sonde (seule la tête est entraînable, réseau et BN en eval même après `train()`, conservé par `deepcopy`, réseau et statistiques BN inchangés après un pas d'optimisation) ; registre ; groupes de paramètres (partition exacte) ; **2 entraînements AMP sur GPU identiques au bit près** par architecture |
| `test_subsets.py` | 16 | 21, 52, 105 et 210 images par classe ; tailles 126, 312 et 630 ; fraction réelle 0,2476 ; emboîtement 10 % ⊂ 25 % ⊂ 50 % pour les 3 graines ; reproductibilité ; indépendance de l'ordre des lignes ; fichiers du train uniquement ; générateur torch non touché ; époque virtuelle de 1 260 échantillons en 40 pas, avec 10, 4 ou 5, et 2 passages par image ; ordre identique au loader historique |
| `test_phase3_pipeline.py` | 10 | pipeline de la phase 2 identique ; réplication en gris exacte ; augmentation commune aux 3 canaux ; colonnes historiques de `history.csv` ; rapport des taux d'apprentissage conservé ; paramètres gelés intacts ; **non-régression : la graine 42 reproduit les 2 premières époques de `custom_cnn/seed_42/history.csv`** (écart ≤ 1e-6, la précision du CSV) ; époques virtuelles sur les vraies données |
| `test_calibration_stats.py` | 13 | NLL, Brier et ECE calculés à la main ; bornes des intervalles ; ECE à effectifs égaux ; Clopper-Pearson égal au tableau du protocole ; McNemar (6 désaccords : p = 0,03125 ; 5 : 0,0625) ; règle NLL dans les 3 issues ; sélection (marge, rappel, domination, latence, départage par taille) |
| `test_benchmark_robustness.py` | 18 | nombre d'itérations de préchauffage et de mesure ; quartiles ; débit ; mode eval ; nombre de threads restauré ; MAC (convolution 3×3 : 900) ; taille d'un `state_dict` ; niveaux d'E5 conformes ; perturbations déterministes et dans [0, 1] ; condition « clean » égale au pipeline standard |
| `test_comparison.py` | 6 | chargement (les runs annexes sont ignorés, l'ordre des images est normalisé) ; refus de `test_used` ; alignement ; aire sous la courbe d'apprentissage ; script complet sur des runs synthétiques (sélection « pending » sans benchmark, refus d'écraser) ; **la NLL recalculée à partir des CSV de la phase 2 égale la loss stockée** (probabilités brutes) |

## 4. Commandes (depuis `ai-engine/`)

```powershell
python -m pytest                                          # 137 tests
python scripts/download_weights.py                        # déjà fait ; ne retélécharge pas si les poids sont en cache
python scripts/train.py --config configs/resnet50_ft.yaml --seed 42
python scripts/train.py --config configs/resnet50_ft.yaml --seed 42 --train-fraction 0.1   # -> resnet50_ft_f010
python scripts/summarize_runs.py --experiment resnet50_ft
python scripts/benchmark_models.py                        # sur secteur
python scripts/evaluate_robustness.py                     # E5, complémentaire
python scripts/compare_models.py
```

## 5. Compatibilité avec les phases 1 et 2

- **Fichiers modifiés** : uniquement du code, de façon rétrocompatible : `ivai/data/dataset.py`, `ivai/training/trainer.py`, `scripts/train.py`. Les configurations, résultats, figures, splits, checkpoints, tests et documents des phases 1 et 2 ne sont pas modifiés.
- **Empreintes SHA-256** des 74 fichiers de `artifacts/`, `ai-engine/reports/` et `ai-engine/datasets/splits/` : identiques avant et après l'implémentation.
- **Changement de comportement volontaire** : `train.py` refuse désormais d'écraser un run existant. Une relance de la phase 2 se fait avec `--tag` (par exemple `seed_42_nonreg`).
- `scripts/summarize_runs.py` (phase 2) fonctionne sans modification sur les nouvelles expériences.

## 6. Écarts au protocole

Aucun écart à ce stade. Précisions d'implémentation, qui n'entrent pas en conflit avec le protocole :
- latence mesurée sur le checkpoint de la graine 42 ;
- JPEG réalisé avec Pillow ;
- sorties dérivées dans `reports/benchmarks/phase3/`, `reports/robustness/phase3/` et `reports/comparison/phase3/`.

## 6 bis. Validation pré-expérimentale (étape 3)

**Poids ImageNet** : téléchargés avec `scripts/download_weights.py` dans le cache torch (`~/.cache/torch/hub/checkpoints`), hors du dépôt et non versionnés. Le SHA-256 complet de chaque fichier commence par le préfixe inscrit dans son nom (vérifié par torchvision avec `check_hash=True`, puis par le script).

| Poids | Fichier | Taille | SHA-256 |
|---|---|---|---|
| `ResNet50_Weights.IMAGENET1K_V1` | `resnet50-0676ba61.pth` | 97,8 Mo | `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a` |
| `MobileNet_V2_Weights.IMAGENET1K_V1` | `mobilenet_v2-b0353104.pth` | 13,6 Mo | `b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889` |

**Non-régression du CNN** : relance complète de la phase 2 avec le code de la phase 3 (`train.py --config configs/custom_cnn.yaml --seed 42 --tag nonreg`, sorties dans `seed_42_nonreg`).
- Le verdict de `scripts/compare_runs.py` contre `seed_42` est **`BIT-EXACT REPRODUCTION`**.
- Historique des 52 époques identique.
- Meilleure époque 44 dans les deux runs.
- Prédictions et probabilités de validation identiques octet pour octet.
- 26 tenseurs sur 26 identiques dans le checkpoint sélectionné.
- Accuracy 0,9963 et perte de validation 0,024517, identiques.
- Seuls diffèrent les métadonnées (date, chemins, champs `data.input` et `steps_per_epoch` ajoutés en phase 3) et le temps d'exécution : 311 s contre 283 s, sur batterie.

## 7. Résultats

À compléter après l'étape 4.
