# Phase 2 : CNN personnalisé (baseline)

Exécution : 14-15/11/2026, RTX A2000 8 GB, PyTorch 2.14.1+cu130. **Toutes les métriques
sont calculées sur la validation (270 images).** Le jeu de test n'a été ni chargé ni évalué.

## 1. Architecture

| Couche | Opération | Sortie (C x H x W) | Paramètres |
|---|---|---|---|
| Entrée | image en niveaux de gris normalisée | 1 x 200 x 200 | – |
| Bloc 1 | Conv 3x3 (32, pad 1, sans biais) → BN → ReLU → MaxPool 2 | 32 x 100 x 100 | 288 + 64 |
| Bloc 2 | Conv 3x3 (64) → BN → ReLU → MaxPool 2 | 64 x 50 x 50 | 18 432 + 128 |
| Bloc 3 | Conv 3x3 (128) → BN → ReLU → MaxPool 2 | 128 x 25 x 25 | 73 728 + 256 |
| Bloc 4 | Conv 3x3 (256) → BN → ReLU → MaxPool 2 | 256 x 12 x 12 | 294 912 + 512 |
| GAP | moyenne spatiale | 256 | 0 |
| Dropout | p = 0,5 (entraînement uniquement) | 256 | 0 |
| Linéaire | 256 → 6 (**logits**, pas de softmax) | 6 | 1 542 |
| **Total** | | | **389 862** (vérifié dans le code et par un test) |

Rôle de chaque couche :
- **Convolution 3x3** : détecte des motifs locaux (bords, points, textures). Elle n'a pas de
  biais car la BatchNorm qui suit en a déjà un (β).
- **BatchNorm** : normalise les activations de chaque mini-batch. L'entraînement est plus
  stable et accepte un taux d'apprentissage plus élevé.
- **ReLU** : introduit la non-linéarité.
- **MaxPool 2x2** : divise la résolution par 2, rend le réseau légèrement invariant aux
  petites translations et réduit le calcul. 25 → 12 car la division est arrondie vers le bas.
- **Global Average Pooling** : résume chaque carte de caractéristiques par sa moyenne. Il
  remplace un « flatten » de 36 864 valeurs : la couche finale passe de 221 190 à 1 542
  paramètres, ce qui limite fortement le sur-apprentissage sur 1 260 images.
- **Dropout 0,5** : désactive aléatoirement la moitié des neurones pendant l'entraînement,
  ce qui régularise le modèle.
- **Logits** : `CrossEntropyLoss` applique déjà log-softmax en interne, de façon
  numériquement stable. La softmax n'est appliquée qu'à l'inférence, pour obtenir les
  probabilités.

**Champ réceptif.** Il est de **46 px** en théorie (formule rₙ = rₙ₋₁ + (k−1)·jₙ₋₁), et la
mesure par rétropropagation du gradient, avec MaxPool remplacé par AvgPool, donne aussi
**46 px**. Avec MaxPool, le gradient ne passe que par les maxima : sur une image donnée, la
zone réellement utilisée était de 38 px. Chaque neurone final voit donc une zone de
46 x 46 px, soit environ 23 % de la largeur de l'image, et le GAP agrège ces zones locales.

## 2. Données et augmentations

- Les CSV de split de la phase 1 sont utilisés tels quels. Un test vérifie que leur SHA-256
  est inchangé.
- Normalisation : moyenne 0,5043, écart-type 0,2125 (train uniquement, phase 1).
- Le **test est verrouillé** : `NEUDataset("test")` lève `TestSetAccessError`.
- Augmentations **sur le train uniquement** (vérifié par test) :

| Augmentation | Justification mesurée (`scripts/analyze_augmentations.py`, train) |
|---|---|
| Retournement horizontal (p = 0,5) | Conserve l'orientation des textures. L'asymétrie gauche/droite significative vient du bord droit sombre, un artefact d'éclairage présent dans **toutes** les classes : le retournement rend le modèle insensible à cet artefact. |
| Retournement vertical (p = 0,5) | Conserve l'orientation. Aucune classe n'a d'asymétrie haut/bas au-delà du bruit (95ᵉ percentile de l'hypothèse nulle). |
| Luminosité et contraste ±20 % | Atténue le raccourci « luminosité moyenne » observé pendant l'EDA. |
| **Pas de rotation de 90°** | rolled-in_scale a 100 % de textures horizontales, les 5 autres classes de 72 à 100 % de textures verticales. Une rotation inverserait cette orientation, qui est une information réelle. |

## 3. Entraînement

Adam (lr 1e-3), CrossEntropyLoss, batch de 32, `ReduceLROnPlateau` (facteur 0,5, patience 3)
sur la loss de validation, early stopping (patience 8), checkpoint du **meilleur modèle selon
la loss de validation**, AMP fp16 avec GradScaler, mode déterministe. Mémoire GPU maximale :
566 Mo.

### Évolution de la configuration, décidée sur la validation uniquement

| Version | Epochs max | Constat | Décision |
|---|---|---|---|
| `custom_cnn_30ep` | 30 | Aucun early stopping ; meilleure epoch = 30 (la dernière) pour 2 graines sur 3 → la limite de 30 epochs coupait l'apprentissage | Passer à 60 epochs, **sans autre changement** |
| `custom_cnn` (retenue) | 60 | Early stopping pour la graine 42 (epoch 44) ; graines 123 et 2026 au meilleur à l'epoch 57 et 59 | Arrêt ici (voir ci-dessous) |

**Pourquoi ne pas aller au-delà de 60 epochs ?** Entre les epochs 40 et 60, la loss de
validation ne baisse plus que très peu (environ 0,03 → 0,025) et l'accuracy est saturée
(1 à 2 erreurs sur 270). Continuer à ajuster le modèle sur ces 270 images ferait surtout
**apprendre la validation elle-même** (biais de sélection). L'architecture n'a pas été
modifiée.

## 4. Résultats (validation, 270 images)

| Seed | Accuracy | Précision macro | Rappel macro | F1 macro | Val loss | Erreurs | Meilleure epoch | Epochs | Early stop | Temps |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0,9963 | 0,9964 | 0,9963 | 0,9963 | 0,0245 | 1 | 44 | 52 | oui | 283 s |
| 123 | 0,9963 | 0,9964 | 0,9963 | 0,9963 | 0,0228 | 1 | 57 | 60 | non | 350 s |
| 2026 | 0,9926 | 0,9929 | 0,9926 | 0,9926 | 0,0279 | 2 | 59 | 60 | non | 424 s* |
| **moyenne ± écart-type** | **0,9951 ± 0,0021** | 0,9952 ± 0,0020 | 0,9951 ± 0,0021 | **0,9951 ± 0,0021** | 0,0251 ± 0,0026 | 1,3 ± 0,6 | 53,3 ± 8,1 | | | 352 ± 70 s |

Écart-type d'échantillon (ddof = 1) sur 3 graines : c'est une indication de la variabilité,
pas un intervalle de confiance. \* La graine 2026 a été relancée sur secteur après une mise en
veille pendant la première exécution ; ses métriques sont identiques au bit près.

Configuration à 30 epochs, pour comparaison : accuracy 0,9914 ± 0,0021, F1 0,9914 ± 0,0021,
val loss 0,0401 ± 0,0067, 2,3 ± 0,6 erreurs.

**Par classe** (moyenne sur les 3 graines) :

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| crazing | 1,000 | 1,000 | 1,000 ± 0,000 |
| inclusion | 0,971 | 1,000 | 0,985 ± 0,006 |
| patches | 1,000 | 1,000 | 1,000 ± 0,000 |
| pitted_surface | 1,000 | 1,000 | 1,000 ± 0,000 |
| rolled-in_scale | 1,000 | 1,000 | 1,000 ± 0,000 |
| scratches | 1,000 | 0,970 | 0,985 ± 0,007 |

**Analyse des erreurs** (`reports/experiments/custom_cnn/val_errors.csv`, figure `val_errors.png`) :
- Les **4 erreurs** des 3 graines finales sont toutes des `scratches` prédites `inclusion`.
- `scratches_44` est mal classée par **les 6 modèles** (30 et 60 epochs), avec une confiance de
  0,91 à 0,94. C'est une rayure verticale très peu contrastée sur fond sombre, visuellement
  proche d'une inclusion. C'est un cas réellement ambigu, pas un bruit d'entraînement.
- `scratches_76` (rayure fine en bord d'image) n'est mal classée que par 1 modèle final sur 3,
  avec une confiance faible (0,61).
- Les **22 images multi-défauts** de la validation sont toutes bien classées.

**Figures** (`ai-engine/reports/figures/custom_cnn/`) : `seed_*_curves.png`,
`seed_*_confusion_val.png`, `summary_val_curves.png`, `summary_f1_per_class.png`,
`summary_confusion_val_sum.png`, `val_errors.png`, `aug_orientation.png`.

## 5. Reproductibilité

Graines fixées (Python, NumPy, Torch, DataLoader), cuDNN déterministe sans benchmark,
`torch.use_deterministic_algorithms(True)` et `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Le GAP est
écrit `x.mean((2,3))` car le backward d'`AdaptiveAvgPool2d` n'est pas déterministe sur CUDA.

| Vérification | Résultat |
|---|---|
| 3 graines à 30 epochs relancées | **identiques au bit près** (historique, probabilités, 26 tenseurs de poids) |
| Graine 42 à 60 epochs, seconde exécution (`seed_42_repro`) | **identique au bit près** |
| Graine 2026 : exécution interrompue par la veille contre relance sur secteur | **identique au bit près** |
| Test unitaire (2 epochs, CPU et GPU) | identique ; une graine différente donne un résultat différent |

**Limite :** la garantie vaut pour une même configuration matérielle et logicielle (même GPU,
pilote et versions). Un autre GPU ou une autre version de cuDNN peut donner des résultats
légèrement différents.

**Ce qui n'est PAS reproductible : le temps.** Pour un calcul identique, la durée d'une epoch
a varié de 4,3 à 16,8 s (graine 42 : 283 s puis 559 s). Les causes sont la fréquence du CPU
(qui fait les augmentations), la température et la charge de la machine. Les temps
d'entraînement sont donc des **ordres de grandeur** (environ 5 à 9 s par epoch). Pour comparer
les architectures, on mesurera la latence d'inférence avec un protocole contrôlé (préchauffage,
répétitions, médiane).

## 6. Tests (49 au total, tous passent)

| Fichier | Vérifie |
|---|---|
| `test_custom_cnn.py` | Forme de sortie (batch x 6), forme des cartes (256x12x12), refus d'une entrée à 3 canaux, logits finis sans softmax, loss et gradients, capacité à sur-apprendre un mini-batch, **389 862 paramètres**, **champ réceptif 46**, cohérence CPU/GPU, sauvegarde et chargement des poids |
| `test_dataset.py` | Test verrouillé, **absence de fuite** (noms, hash de pixels, groupes de doublons), CSV de split inchangés, formes, normalisation (moyenne 0 et écart-type 1 sur le train), aucune augmentation hors train, aucune rotation |
| `test_training.py` | Early stopping, **reproductibilité exacte** CPU et GPU, graines différentes donnant des résultats différents |

## 7. Commandes de reproduction (depuis `ai-engine/`)

```powershell
python -m pytest                                                     # 49 tests
python scripts/analyze_augmentations.py                              # justification des flips (train)
python scripts/train.py --config configs/custom_cnn.yaml --seed 42
python scripts/train.py --config configs/custom_cnn.yaml --seed 123
python scripts/train.py --config configs/custom_cnn.yaml --seed 2026
python scripts/summarize_runs.py --experiment custom_cnn             # moyenne ± écart-type + figures
python scripts/plot_val_errors.py --experiment custom_cnn --experiment custom_cnn_30ep
# vérification de reproductibilité
python scripts/train.py --config configs/custom_cnn.yaml --seed 42 --tag repro
python scripts/compare_runs.py reports/experiments/custom_cnn/seed_42 reports/experiments/custom_cnn/seed_42_repro ../artifacts/models/custom_cnn/seed_42/best.pt ../artifacts/models/custom_cnn/seed_42_repro/best.pt
```

À lancer **sur secteur**. Le script empêche lui-même la mise en veille pendant l'entraînement.

## 8. Difficultés rencontrées

| Problème | Cause | Solution |
|---|---|---|
| Champ réceptif mesuré à 38 px contre 46 px en théorie | Le backward du MaxPool ne passe que par le maximum | Mesure géométrique avec AvgPool (46 = théorie) ; les deux valeurs sont documentées |
| Limite de 30 epochs atteinte sans convergence | Meilleure epoch = dernière epoch pour 2 graines sur 3 | Passage à 60 epochs ; la version à 30 epochs est conservée |
| Temps de la graine 2026 : 10 148 s | Mise en veille Windows pendant l'epoch 60 (événements Kernel-Power 506/507) | Relance sur secteur, résultats identiques au bit près ; ajout de `prevent_sleep()` dans `train.py` |
| Seconde exécution extrêmement lente | Ordinateur sur batterie : GPU bridé par la limite de puissance | Exécution arrêtée, puis relancée sur secteur |
| Temps variable du simple au double pour un même calcul | Charge, température et fréquence du CPU du portable | Temps présentés comme ordres de grandeur ; protocole contrôlé prévu pour la latence |
| Backward d'`AdaptiveAvgPool2d` non déterministe sur CUDA | Implémentation PyTorch | GAP écrit `x.mean((2,3))`, mathématiquement identique |

## 9. Points clés pour la soutenance

1. **La validation sert à choisir, le test à mesurer.** Toutes les décisions (nombre
   d'epochs, epoch retenue) ont été prises sur la validation. Le test reste intact pour une
   mesure finale non biaisée.
2. **Pourquoi la loss de validation est sous celle du train ?** Le train est évalué avec le
   dropout actif et des augmentations ; la validation est évaluée en mode `eval`, sans
   dropout ni augmentation, et la BatchNorm y utilise ses statistiques globales. Ce n'est pas
   une fuite.
3. **Pourquoi des pics de validation au début (epoch 11 de la graine 42) ?** Avec lr = 1e-3, les
   statistiques globales de la BatchNorm sont encore instables. Ces pics disparaissent quand
   `ReduceLROnPlateau` divise le taux d'apprentissage.
4. **Un modèle de 390 000 paramètres atteint 99,5 % sur la validation** : NEU-CLS est un dataset
   relativement facile et équilibré, sur lequel la littérature rapporte des scores
   comparables. La comparaison avec ResNet50 (environ 25 M de paramètres) devra donc porter
   autant sur le coût (taille, latence) que sur l'accuracy.
5. **Avec 270 images, une erreur vaut 0,37 point d'accuracy.** L'écart entre 99,26 % et 99,63 %
   correspond à **une seule image** : il n'est pas significatif.
6. **L'erreur systématique `scratches_44` → `inclusion`** illustre une limite : une rayure peu
   contrastée ressemble à une inclusion. Grad-CAM (phase 6) permettra de voir sur quelles
   zones le modèle s'appuie.
