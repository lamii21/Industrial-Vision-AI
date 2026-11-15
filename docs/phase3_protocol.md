# Phase 3 : protocole expérimental (gelé)

> **Statut : GELÉ le 15/11/2026, avant toute implémentation et tout entraînement.**
> Ce document fixe à l'avance les expériences, les configurations, les métriques et les
> critères de sélection de la phase 3. Il ne sera **pas modifié après observation des
> résultats**. Tout écart nécessaire (bug, contrainte matérielle) sera consigné dans la section
> « Écarts au protocole » de `docs/phase3.md`, avec sa justification et son impact, sans réécrire
> ce document.
>
> Point de départ : commit `899e66c` (phase 2). Le jeu de **test reste verrouillé** pendant toute
> la phase 3.

---

## 1. Contexte : constats des phases 1 et 2

| Élément | Constat | Source |
|---|---|---|
| Tâche | Classification de 6 défauts de surface d'acier (NEU-CLS) | phase 1 |
| Données | 1 800 images 200×200 en niveaux de gris, **6 × 300 (classes équilibrées)** | phase 1 |
| Split | Stratifié par groupes de doublons, 70/15/15, graine 42 : train 1 260 / val 270 / test 270 (210/45/45 par classe), 8/8 contrôles anti-fuite | phase 1 |
| Particularités | 123 images multi-défauts (6,8 %) ; écarts de luminosité entre classes ; bord droit sombre ; textures orientées | phase 1 (EDA) |
| Baseline | CNN personnalisé de 4 blocs, **389 862** paramètres, champ réceptif 46 px | phase 2 |
| Résultat | Validation : accuracy **0,9951 ± 0,0021**, F1 macro 0,9951 ± 0,0021, NLL 0,0251 ± 0,0026, 1 à 2 erreurs sur 270 | phase 2 |
| Erreur systématique | `scratches_44` → `inclusion` pour les 6 modèles entraînés, avec une confiance de 0,91 à 0,94 | phase 2 |

**Conséquence centrale : la validation est saturée.** Avec n = 270, une erreur vaut 0,37 point
d'accuracy. Une comparaison fondée seulement sur l'accuracy à 100 % des données ne peut départager
aucun modèle (voir §11). La phase 3 compare donc les modèles sur plusieurs axes : performance,
qualité probabiliste, efficacité en données et coût. **Elle ne cherche pas à améliorer un score.**

## 2. Objectifs

- **O1.** Intégrer ResNet50 et MobileNetV2, préentraînés sur ImageNet, sur **le même protocole**
  que la baseline : mêmes données, mêmes augmentations, mêmes graines, même critère de
  checkpoint, même validation.
- **O2.** Mesurer l'effet du **préentraînement** ImageNet séparément de l'effet de
  l'**architecture** (ablation).
- **O3.** Comparer les modèles sur la **performance**, la **qualité probabiliste**,
  l'**efficacité en données** et le **coût de déploiement**.
- **O4.** Choisir le modèle par défaut de l'API d'inférence selon les critères du §10, **fixés à
  l'avance**. Les trois modèles du projet resteront disponibles dans le module de comparaison.

## 3. Hypothèses et questions

| # | Énoncé | Expérience | Nature |
|---|---|---|---|
| H1 | À 100 % des données, aucune différence d'accuracy **détectable** entre les trois modèles (saturation) | E1 | hypothèse ; un résultat nul est attendu et sera rapporté comme tel |
| H2 | Le préentraînement améliore l'**efficacité en données** : à 10 % du train, les modèles fine-tunés dépassent le CNN personnalisé et MobileNetV2 entraîné depuis zéro | E2, E3 | hypothèse |
| H3 | Les caractéristiques ImageNet gelées (sonde linéaire) sont moins bonnes que le fine-tuning, à cause de l'écart de domaine (images naturelles en couleur contre textures d'acier en niveaux de gris) | E2 | hypothèse |
| Q1 | Quel modèle a la plus faible latence CPU ? Le CNN personnalisé demande 2,2 fois plus de calcul que MobileNetV2, mais utilise des opérations plus simples | E4 | question ouverte |
| Q2 | Les modèles préentraînés sont-ils moins sur-confiants sur leurs erreurs ? | E1 | question exploratoire |
| Q3 | Quelle robustesse face à des perturbations d'acquisition hors de la plage d'augmentation ? | E5 | question exploratoire, complémentaire |

## 4. Données

- **Split inchangé.** Les CSV de la phase 1 sont utilisés tels quels et leur SHA-256 est vérifié
  par les tests existants.
- **Test verrouillé.** `NEUDataset("test")` lève `TestSetAccessError`. Aucun script de la phase 3
  ne passe `allow_test=True`.
- **Validation** : les 270 images, complètes, pour **toutes** les expériences, y compris E3.
- **Augmentations** (train uniquement), identiques à la phase 2 pour tous les modèles :
  - flip horizontal p = 0,5 ;
  - flip vertical p = 0,5 ;
  - luminosité et contraste ±20 % ;
  - aucune rotation de 90°.
- **Entrée** : résolution native 200×200, sans redimensionnement à 224. Le redimensionnement
  interpolerait sans ajouter d'information, et les réseaux à pooling global acceptent cette
  taille.
- **Normalisation** :
  - CNN personnalisé : 1 canal, moyenne 0,5043 et écart-type 0,2125 (train, phase 1) ;
  - modèles torchvision, préentraînés ou non : canal gris **répliqué sur 3 canaux**, puis
    normalisation ImageNet (moyenne 0,485/0,456/0,406, écart-type 0,229/0,224/0,225). Les poids
    préentraînés supposent ces statistiques. MobileNetV2 entraîné depuis zéro reçoit la même
    entrée que son jumeau préentraîné, si bien que seul le préentraînement diffère.

### Sous-ensembles d'E3

- Tirés **du train uniquement**, stratifiés par classe et **emboîtés** : pour chaque classe, une
  permutation des 210 images du train est tirée et on garde ses k premières.
- Valeurs de k :

| Fraction annoncée | Images par classe | Total | **Fraction réelle** |
|---|---|---|---|
| 10 % | 21 | 126 | 0,1000 |
| 25 % | **52** | **312** | **0,2476** (52,5 n'est pas entier ; arrondi inférieur) |
| 50 % | 105 | 630 | 0,5000 |

- La permutation dépend de la **graine du run** (42, 123 ou 2026), pas de l'architecture. À graine
  égale, les trois architectures voient donc **exactement le même sous-ensemble**, ce qui permet
  une comparaison appariée. L'écart-type entre graines inclut la variabilité due à
  l'échantillonnage des données.
- La liste des fichiers de chaque sous-ensemble est sauvegardée en CSV dans le dossier du run.
- La normalisation reste celle du train complet : ce sont des statistiques du train, sans aucun
  lien avec la validation ou le test.

## 5. Architectures

Nombres de paramètres **mesurés** sur les modèles torchvision 0.29.1 (`weights=None`) après
remplacement de la dernière couche (étape 1) :

| Modèle | Poids initiaux | Paramètres (6 classes) | Tête | Calcul à 200×200 | Carte finale | Taille fp32 |
|---|---|---|---|---|---|---|
| CNN personnalisé (phase 2) | aucun | **389 862** | GAP, Dropout 0,5, Linear 256→6 (1 542) | 0,564 GMAC | 256×12×12 | 1,5 Mo |
| ResNet50 | ImageNet `IMAGENET1K_V1` | **23 520 326** | `fc` : Linear 2048→6 (12 294) | 3,497 GMAC | 2048×7×7 | 89,9 Mo |
| MobileNetV2 | ImageNet `IMAGENET1K_V1` | **2 231 558** | `classifier` : Dropout 0,2, Linear 1280→6 (7 686) | 0,261 GMAC | 1280×7×7 | 8,6 Mo |

Précisions :
- Les valeurs de référence torchvision (25 557 032 et 3 504 872) incluent la tête à 1 000 classes
  (2 049 000 et 1 281 000 paramètres).
- Le calcul est mesuré par `torch.utils.flop_counter` (convolutions et produits matriciels) et
  exprimé en multiplications-additions (MAC).
- Le choix des poids **V1** est fait à l'avance : ce sont les poids de référence de la littérature
  (He et al., 2016 ; Sandler et al., 2018). Les poids V2, entraînés avec une recette plus lourde,
  ne sont pas utilisés.
- La tête d'origine de torchvision est conservée et seule sa dernière couche linéaire est
  remplacée, sans aucune modification de l'architecture.
- **Pooling global** : `AdaptiveAvgPool2d` est conservé tel quel (voir §13.2).

**Architectures écartées** :
- EfficientNet-B0 : même famille de blocs que MobileNetV2, sans question scientifique nouvelle ;
- ViT et transformers : exigent beaucoup de données, mal adaptés à 1 260 images ;
- ResNet18 : absent du cahier des charges.

## 6. Configurations d'entraînement

Aucune recherche d'hyperparamètres n'est faite sur la validation, pour aucun modèle. La baseline
n'a pas été optimisée, les autres modèles ne le seront pas non plus. Les valeurs suivent la
pratique standard, fixée a priori.

| Réglage | CNN perso (réutilisé) | Fine-tuning (FT) | MobileNetV2 depuis zéro | Sonde linéaire |
|---|---|---|---|---|
| Paramètres entraînés | tous | tous | tous | **dernière couche seule** ; réseau gelé (`requires_grad=False`) et maintenu **en mode eval** (statistiques BatchNorm figées) |
| Optimiseur | Adam | Adam | Adam | Adam |
| Taux d'apprentissage | 1e-3 | **réseau 1e-4, dernière couche 1e-3** (deux groupes) | 1e-3 | 1e-3 |
| Weight decay | 0 | 0 | 0 | 0 |
| Scheduler | ReduceLROnPlateau sur la loss de validation, facteur 0,5, patience 3, min 1e-6 | identique, appliqué **à chaque groupe** | identique | identique |
| Arrêt anticipé | patience 8, min_delta 0 | identique | identique | identique |
| Checkpoint | minimum de la loss de validation | identique | identique | identique |
| Budget | 60 époques maximum, 40 pas par époque | identique | identique | identique |
| Batch, précision | 32, AMP fp16 avec GradScaler | identique | identique | identique |
| Déterminisme | graines, cuDNN déterministe, `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | identique | identique | identique |
| Graines | 42, 123, 2026 | identiques | identiques | identiques |

### Nommage et emplacement des sorties

| Configuration (`ai-engine/configs/`) | Expérience |
|---|---|
| `custom_cnn.yaml` (existante, **non modifiée**) | `custom_cnn` |
| `resnet50_ft.yaml` | `resnet50_ft` |
| `mobilenetv2_ft.yaml` | `mobilenetv2_ft` |
| `mobilenetv2_scratch.yaml` | `mobilenetv2_scratch` |
| `resnet50_probe.yaml` | `resnet50_probe` |
| `mobilenetv2_probe.yaml` | `mobilenetv2_probe` |
| E3 : mêmes configurations avec `--train-fraction` | `<expérience>_f010`, `_f025`, `_f050` |

Les sorties gardent la même structure qu'en phase 2 :
- `reports/experiments/<exp>/seed_<N>/` : `run.json`, `history.csv`, `val_predictions.csv`, et pour
  E3 `train_subset.csv` ;
- `artifacts/models/<exp>/seed_<N>/best.pt` : poids, **non versionnés**.

**Aucun écrasement.** Le script refusera de démarrer si le dossier du run ou le checkpoint existe
déjà. Les dossiers des phases 1 et 2 ne sont jamais écrits. `run.json` enregistre en plus :
- le nom et l'URL des poids préentraînés ;
- la normalisation utilisée ;
- la fraction annoncée et la fraction réelle ;
- les taux d'apprentissage par groupe.

## 7. Expériences

| # | Expérience | Configurations | Graines | Runs nouveaux | Statut |
|---|---|---|---|---|---|
| **E1** | Comparaison à 100 % du train | `resnet50_ft`, `mobilenetv2_ft`, plus `custom_cnn` (phase 2, **réutilisé sans réentraînement**) | 3 | 6 | obligatoire |
| **E2** | Ablation du préentraînement | `mobilenetv2_scratch`, `resnet50_probe`, `mobilenetv2_probe` | 3 | 9 | obligatoire |
| **E3** | Efficacité en données | fractions 10, 25 et 50 % pour `custom_cnn`, `resnet50_ft` et `mobilenetv2_ft` ; le point à 100 % vient d'E1 | 3 | 27 | obligatoire |
| **E4** | Coût et déploiement | checkpoints d'E1 et d'E2 | – | 0 | obligatoire |
| E5 | Robustesse | modèles d'E1 sur la validation perturbée (§9.6) | – | 0 | complémentaire, si les ressources le permettent |

Les métriques de la baseline qui n'existaient pas en phase 2 (calibration, NLL par image, coût)
sont calculées à partir des fichiers existants (`val_predictions.csv`, `best.pt`), **en lecture
seule**.

**Total : 42 runs d'entraînement**, plus 1 relance de non-régression (§14).

## 8. Époques virtuelles (E3)

**Problème.** Avec 10 % des données (126 images), une époque ne compte que 4 pas : 60 époques
feraient 240 pas, et une patience de 8 époques ne représenterait que 32 pas. Les règles d'arrêt
n'auraient plus le même sens d'une fraction à l'autre.

**Définition.** Une **époque virtuelle** contient **1 260 échantillons**, autant que le train
complet, soit 40 pas de batch 32, quelle que soit la fraction. Patience, scheduler et nombre
maximal d'époques sont donc **strictement identiques** pour toutes les fractions et toutes les
architectures : 40 pas par époque, **2 400 pas au maximum**.

**Mécanisme.**
- On utilise `RandomSampler(sous_ensemble, num_samples=1260, replacement=False, generator=g)`.
- Une époque enchaîne ⌊1260/n⌋ permutations complètes du sous-ensemble, puis les
  `1260 mod n` premiers éléments d'une permutation supplémentaire.
- Chaque image est donc vue autant de fois que les autres, à une près, dans chaque époque.

| Fraction | n | Passages par image et par époque virtuelle |
|---|---|---|
| 10 % | 126 | exactement 10 (vérifié : 50 sur 5 époques) |
| 25 % | 312 | 4 ou 5 (moyenne 4,04) |
| 50 % | 630 | exactement 2 |
| 100 % | 1 260 | exactement 1 |

**Réutilisation des images.**
- Une même image apparaît plusieurs fois par époque à faible fraction. Chaque passage reçoit une
  augmentation aléatoire indépendante (flips, luminosité, contraste).
- Les combinaisons de flips ne sont que 4 : les répétitions sont donc proches les unes des autres.
- Le **risque de sur-apprentissage augmente quand la fraction diminue**. C'est l'effet qu'E3
  mesure, pas un biais du protocole. Il est commun aux trois architectures, qui ont le même budget
  et les mêmes sous-ensembles.

**Compatibilité avec la phase 2.**
- Pour une fraction de 1,0, cet échantillonneur produit **exactement** le même ordre que le
  `shuffle=True` historique (vérifié sur 4 époques, graine 42 ; étape 1).
- Par prudence, le code conservera malgré tout la construction historique quand la fraction
  vaut 1.

## 9. Métriques

Toutes les métriques sont calculées **sur la validation**, pour le checkpoint retenu, et
présentées en moyenne ± écart-type (ddof = 1) sur 3 graines.

### 9.1 Classification
- Accuracy, précision, rappel et F1 macro.
- Précision, rappel et F1 par classe.
- Matrice de confusion et nombre d'erreurs sur 270.
- Accuracy séparée sur les images à un défaut et sur les images multi-défauts.
- Analyse qualitative des erreurs, avec le suivi de `scratches_44`. Cette analyse est **purement
  descriptive et n'intervient jamais dans la sélection**.

### 9.2 Qualité probabiliste (métriques de référence)
- **NLL** : moyenne de −log p(y|x). Elle est égale à la loss de validation du checkpoint retenu.
- **Score de Brier** multiclasse.
- **Probabilités brutes** : softmax en fp32 des logits du modèle (calculés en fp16 sous AMP),
  **sans ajustement de température ni autre recalibration sur la validation**. Un recalibrage
  ajusté sur la validation rendrait ces métriques optimistes et ferait de la validation un
  ensemble d'apprentissage. Cohérence vérifiée à l'étape 1 : la NLL recalculée à partir de
  `val_predictions.csv` est égale à la loss de validation de la phase 2.
- **Confiance de chaque erreur**, et nombre d'erreurs dont la confiance dépasse 0,9. Ce sont les
  erreurs les plus dangereuses pour un opérateur.

### 9.3 Calibration (descriptive uniquement)

ECE à 15 intervalles de largeur égale, accompagnée :
- de son intervalle bootstrap à 95 % ;
- de la **valeur de bruit** d'un modèle parfaitement calibré (étiquettes tirées selon ses propres
  confiances, 2 000 tirages) ;
- de l'ECE à intervalles d'effectifs égaux ;
- de l'occupation des intervalles ;
- d'un diagramme de fiabilité.

**L'ECE n'intervient pas dans la sélection** (limites au §11.4).

### 9.4 Comparaisons appariées
- Test de McNemar exact (`scipy.stats.binomtest`) sur les prédictions image par image, graine par
  graine, avec une matrice d'accord des erreurs. Il est **rapporté à titre indicatif**, car sa
  puissance est insuffisante (§11.2).
- Différences de NLL appariées par graine et par image (§10).

### 9.5 Coût et déploiement (E4)
- Paramètres (total et entraînables), taille du `state_dict` en fp32 (Mo), calcul en GMAC à
  200×200, mémoire GPU maximale à l'entraînement.
- Temps d'entraînement, présenté **comme un ordre de grandeur** : la phase 2 a montré une variation
  du simple au double pour un calcul identique.
- **Latence**, avec un protocole contrôlé :
  - modèle en mode `eval` sous `torch.inference_mode()` ;
  - entrée fixe à la forme attendue par le modèle ; le prétraitement est exclu et mesuré à part ;
  - **CPU** (Intel i7-12800H) : fp32, `torch.set_num_threads(4)` fixé et enregistré, batch 1 et
    batch 32 ;
  - **GPU** (RTX A2000) : fp32 et AMP fp16, avec `torch.cuda.synchronize()`, batch 1 et 32 ;
  - préchauffage de 50 itérations, puis 200 mesures (10 et 50 pour le batch 32 sur CPU) ;
  - on rapporte la **médiane et l'IQR**, ainsi que le débit en images par seconde ;
  - le benchmark complet est répété 3 fois en alternant l'ordre des modèles, et on rapporte la
    médiane des médianes ;
  - machine **sur secteur**, applications lourdes fermées.

### 9.6 Robustesse (E5, complémentaire)

Les perturbations sont appliquées à la validation (intensités dans [0, 1], avant normalisation),
de façon déterministe : graine fixe par image. Les niveaux sont **en dehors** de la plage
d'augmentation de l'entraînement (±20 %).

| Perturbation | Niveaux |
|---|---|
| Luminosité (facteur) | 0,6 ; 1,4 |
| Contraste (facteur) | 0,6 ; 1,4 |
| Bruit gaussien (σ) | 0,02 ; 0,05 |
| Flou gaussien (σ, noyau 2⌈3σ⌉+1) | 1,0 ; 2,0 |
| Compression JPEG (qualité) | 50 ; 20 |

Métriques : accuracy, F1 macro et NLL pour chaque perturbation, ainsi que leur écart avec la
validation non perturbée. **Purement descriptif, sans rôle dans la sélection.**

### 9.7 Efficacité en données (E3)
- F1 macro, accuracy et NLL en fonction de la fraction réelle (10 %, 24,76 %, 50 %, 100 %).
- Aire sous la courbe d'apprentissage (règle des trapèzes, abscisse = fraction réelle).
- Différences appariées entre architectures à graine et sous-ensemble égaux, selon la même règle
  qu'au §10.2, présentées de façon **descriptive**.

## 10. Critères de sélection du modèle par défaut

**Candidats** : les trois modèles du projet entraînés à 100 % (E1), à savoir `custom_cnn`,
`resnet50_ft` et `mobilenetv2_ft`. Les configurations d'E2 sont des ablations explicatives et ne
sont pas candidates.

### 10.1 Étape 1 : éligibilité
Un candidat est éligible si les deux conditions suivantes sont remplies :
- son **nombre moyen d'erreurs** (moyenne sur 3 graines) est **≤ min_candidats + 2** ;
- **chaque classe** a un **rappel moyen ≥ 0,95**.

### 10.2 Étape 2 : comparaison appariée des NLL

Pour deux candidats éligibles A et B, et pour chaque graine s ∈ {42, 123, 2026} :

- NLL par image : ℓ_{m,s,i} = −log p_{m,s}(y_i | x_i), à partir de `val_predictions.csv`
  (probabilités brutes) ;
- **différence par graine** : d_s = NLL_{A,s} − NLL_{B,s} ;
- **différence par image** : Δ_i = (1/3) Σ_s (ℓ_{A,s,i} − ℓ_{B,s,i}). Sa moyenne sur les 270
  images vaut d̄ = (1/3) Σ_s d_s ;
- **bootstrap apparié sur les images** : 10 000 tirages avec remise des 270 valeurs Δ_i
  (générateur `numpy.random.default_rng(0)`), dont on tire l'intervalle percentile à 95 %
  [L, U] de d̄.

**Règle.** A est **nettement meilleur** que B en NLL (noté A ≻ B) si et seulement si les deux
conditions suivantes sont réunies :

1. **d_s < 0 pour les trois graines** : l'amélioration se répète quel que soit l'entraînement ;
2. **U < 0** : l'intervalle bootstrap apparié par image exclut 0.

Sinon, les NLL de A et B sont dites **proches**. La relation est symétrique : B ≻ A si d_s > 0
pour les trois graines et L > 0.

**Pourquoi deux niveaux d'appariement :**
- **Par image** : les deux modèles sont évalués sur les mêmes 270 images, ce qui élimine la
  variance due à la difficulté propre à chaque image. Mais cet intervalle ignore la variabilité de
  l'entraînement.
- **Par graine** : la condition de signe sur les trois graines capte la variabilité de
  l'entraînement. Seule, elle n'a aucune puissance : sous l'hypothèse nulle, trois signes
  identiques arrivent avec une probabilité de 1/4.
- Exiger les deux conditions évite chacune de ces faiblesses. **On ne compare pas les écarts-types
  individuels des modèles** : ils mélangent la variabilité commune (mêmes images, même graine) et
  surestiment le bruit de la différence.
- Limite de l'appariement par graine : à 100 % des données, il est **faible**. Deux architectures
  avec la même graine ne partagent ni l'initialisation ni, exactement, la suite d'augmentations,
  puisque le générateur aléatoire est consommé différemment. Il est **fort** en E3 (même
  sous-ensemble).

**Contrôle de la règle sur les données de la phase 2** (étape 1, validation, lecture seule) :
- **Même modèle, graines différentes** (aucune différence réelle attendue) : les intervalles
  bootstrap par image de 42−123, 42−2026 et 123−2026 contiennent **tous** 0.
- **60 époques contre 30 époques** (différence réelle) : d_s = −0,0089 ; −0,0239 ; −0,0121 et
  intervalle [−0,0223 ; −0,0092]. La règle conclut à « nettement meilleur ».
- Ce contrôle montre que la règle se comporte raisonnablement. Il **ne garantit pas** un taux de
  faux positifs contrôlé : c'est une règle de décision, pas un test d'hypothèse calibré.

### 10.3 Étape 3 : ensemble des candidats non dominés
- Soit E l'ensemble des candidats éligibles. On retire de E tout candidat B pour lequel il existe
  un A ∈ E tel que A ≻ B. On obtient l'ensemble C.
- C n'est jamais vide. A ≻ B implique d̄(A,B) < 0, et d̄ est additif
  (d̄(A,B) + d̄(B,C) = d̄(A,C)), donc la relation ≻ ne peut pas former de cycle.

### 10.4 Étape 4 : coût
- Dans C, on choisit le modèle à la **plus faible latence CPU médiane au batch 1** (fp32,
  4 threads, §9.5).
- Si l'écart relatif entre les deux meilleures médianes est **inférieur à 5 %**, ce qui est de
  l'ordre de la variabilité de mesure d'un portable, on choisit le modèle au plus petit fichier
  fp32.

### 10.5 Informations complémentaires (jamais décisives)
E3, E5, ECE, analyse des erreurs et McNemar sont rapportés pour éclairer la discussion. Ils **ne
modifient pas** la décision des étapes 1 à 4.

## 11. Limites statistiques

### 11.1 Résolution
Avec n = 270, une erreur vaut 0,37 point. Intervalles exacts de Clopper-Pearson à 95 % sur le
nombre d'erreurs :

| Erreurs observées | Intervalle à 95 % (en images) |
|---|---|
| 0 | [0 ; 3,7] |
| 1 | [0 ; 5,5] |
| 2 | [0,2 ; 7,2] |
| 3 | [0,6 ; 8,7] |

### 11.2 Puissance du test de McNemar
Le test exact ne peut atteindre p < 0,05 qu'avec **au moins 6 désaccords, tous dans le même sens**
(p = 0,031 ; avec 5, p = 0,063). La baseline fait 1 à 2 erreurs : **aucune différence d'accuracy
ne pourra être déclarée significative à 100 % des données.**

### 11.3 Marge d'équivalence de 2 images
- **Justification** : la variabilité propre d'un même modèle entre graines couvre 1 image (1, 1, 2
  erreurs pour la baseline ; 2, 2, 3 pour la version à 30 époques). Une marge de 2 images vaut
  deux fois cette variabilité, appliquée à une moyenne sur 3 graines. Elle correspond aussi à la
  résolution de l'outil.
- **Limites** :
  - ce n'est **pas un test statistique d'équivalence**. Une marge fondée sur les intervalles du
    §11.1 dépasserait 5 images et ne départagerait jamais rien ;
  - un test d'équivalence formel (TOST) est impossible avec n = 270 ;
  - deux modèles jugés « équivalents » peuvent différer réellement d'environ 2 points ;
    inversement, un écart de 3 images peut n'être que du bruit ;
  - moyenner 3 graines réduit la variabilité de l'**entraînement**, pas celle de
    l'**échantillon** : ce sont toujours les mêmes 270 images, et `scratches_44` échoue pour tous
    les modèles de la phase 2.
- La marge est donc une **règle de décision fixée à l'avance**. Elle reporte volontairement le
  choix sur la NLL et le coût, qui sont mieux mesurables.

### 11.4 Limites de l'ECE avec 270 images et 15 intervalles
Mesuré sur la baseline de la phase 2 :

| Graine | ECE (15 intervalles) | Bootstrap 95 % | Bruit d'un modèle parfaitement calibré (médiane / 95ᵉ centile) | Intervalles non vides |
|---|---|---|---|---|
| 42 | 0,0122 | [0,009 ; 0,021] | 0,0094 / 0,0166 | 4 sur 15 |
| 123 | 0,0099 | [0,008 ; 0,019] | 0,0098 / 0,0168 | 4 sur 15 |
| 2026 | 0,0105 | [0,004 ; 0,019] | 0,0110 / 0,0182 | 5 sur 15 |

1. Le dernier intervalle [0,933 ; 1] contient 95 à 97 % des images : l'ECE se réduit
   pratiquement à **un seul intervalle**.
2. **L'ECE observée se situe dans le bruit d'un modèle parfaitement calibré** : elle ne permet pas
   de conclure à une mauvaise calibration.
3. La valeur dépend du découpage : l'ECE à effectifs égaux donne 0,0059 à 0,0094.
4. Une moyenne masque les phénomènes opposés. La baseline est globalement **sous-confiante**
   (confiance moyenne ≈ 0,987 pour une accuracy de 0,995) mais **sur-confiante sur son erreur
   systématique** (0,91 à 0,94).

D'où la décision : NLL et Brier comme métriques de référence, ECE seulement descriptive.

### 11.5 Limites de la NLL
- Avec très peu d'erreurs, la NLL est **dominée par quelques images**. Pour la baseline,
  `scratches_44` représente à elle seule **39 à 45 %** de la NLL, les 3 pires images 55 à 61 %,
  les 10 pires 65 à 74 %.
- Sélectionner sur la NLL revient donc en grande partie à **préférer le modèle qui se trompe avec
  le moins d'assurance**. C'est cohérent avec le principe de conformité (une erreur sûre d'elle est
  la plus trompeuse pour l'opérateur), mais le résultat dépend de quelques images.
- Le bootstrap par image du §10.2 reflète cette dépendance : les intervalles s'élargissent quand
  quelques images dominent.

### 11.6 Graines
- Trois graines donnent une **indication** de la variabilité, pas un intervalle de confiance.
- Le déterminisme garantit la reproductibilité sur la même configuration matérielle et logicielle
  (GPU, pilote, versions), pas au-delà.

## 12. Biais de sélection sur la validation et usage du test

**La validation joue trois rôles dans cette phase :**
1. choisir le checkpoint de chaque run (minimum de la loss de validation sur jusqu'à 60 époques) ;
2. piloter l'entraînement (scheduler et arrêt anticipé réagissent à la loss de validation) ;
3. comparer les modèles et choisir le modèle par défaut (§10).

**Conséquences, à garder en tête pour toute interprétation :**
- **Les résultats de validation ne sont pas une estimation indépendante de la performance en
  généralisation.** Ils sont **biaisés à la hausse** :
  - prendre le minimum sur environ 60 époques favorise une époque « chanceuse » ;
  - choisir le meilleur parmi plusieurs modèles amplifie cet optimisme (« malédiction du
    gagnant »).
- Ce biais existait déjà en phase 2 : le choix de passer de 30 à 60 époques a été fait sur la
  validation.
- Les mesures de validation servent **à comparer et à décider**, pas à annoncer une performance.
- Les comparaisons relatives restent informatives, car tous les modèles subissent le même
  mécanisme de sélection. Mais **aucun chiffre de validation ne sera présenté comme la performance
  attendue en production.**

**Usage du test :**
- Le test (270 images) reste **verrouillé** pendant toute la phase 3.
- Il sera utilisé **une seule fois**, pendant la phase d'évaluation finale, **après le gel** de
  toutes les architectures, configurations, checkpoints et décisions, y compris le choix du modèle
  par défaut.
- Tous les modèles gelés y seront évalués et **tous les résultats seront rapportés**, qu'ils soient
  favorables ou non. Aucune décision ne sera revue après cette évaluation.
- Les limites de résolution du §11 s'appliquent aussi au test (n = 270).

## 13. Vérifications préalables (étape 1, 15/11/2026)

Environnement : Python 3.12.8, torch 2.14.1+cu130, torchvision 0.29.1+cu130, CUDA 13.0,
cuDNN 9.24, NVIDIA RTX A2000 8 GB Laptop, Intel i7-12800H (14 cœurs, 20 threads).

### 13.1 Architectures
Nombres de paramètres et calcul du §5, mesurés sans télécharger de poids (`weights=None`).

### 13.2 Déterminisme d'`AdaptiveAvgPool2d`

| Test | Sortie 1×1 (ResNet50, MobileNetV2) | Sortie 2×2 (contrôle) |
|---|---|---|
| Opérations exécutées (forward + backward) | `mean`, `expand`, `div` | `_adaptive_avg_pool2d` et son backward |
| `use_deterministic_algorithms(True)` (fp32 et fp16, contigu et channels_last) | aucune erreur | erreur : « adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation » |
| 50 backward répétés, comparaison bit à bit | 0/49 différences | 0/49 |
| 5 pas d'entraînement AMP lancés deux fois | ResNet50 : 0/320 tenseurs différents ; MobileNetV2 : 0/314 | – |

**Conclusion** : pour une sortie 1×1, PyTorch calcule une moyenne
(`mean`), déterministe. **Aucun remplacement n'est nécessaire.** Des tests vérifieront que
l'opération exécutée reste `mean` et que deux entraînements courts sont identiques au bit près,
pour chaque architecture.

### 13.3 Échantillonneur
- Pour une fraction de 1,0, l'ordre est identique au `shuffle=True` historique (4 époques).
- À 10 %, chaque image est vue 10 fois par époque virtuelle.

### 13.4 Budget mesuré
Tenseurs aléatoires, 40 pas, AMP, mode déterministe. Mesures faites **sur batterie**, donc
pessimistes.

| Modèle | GPU par époque | Passe de validation | Mémoire GPU max |
|---|---|---|---|
| CNN personnalisé | 1,8 s | 0,13 s | 566 Mo |
| MobileNetV2 | 3,8 s | 0,21 s | 1 113 Mo |
| ResNet50 | 7,2 s | 0,55 s | 1 550 Mo |

## 14. Non-régression du CNN personnalisé

Les modifications du trainer (groupes de paramètres, gel, époques virtuelles) ne doivent pas
changer le comportement historique :

1. **Tests automatiques** :
   - avec une configuration de phase 2, le trainer crée un seul groupe de paramètres au taux
     d'apprentissage historique ;
   - l'ordre des données est identique pour une fraction de 1,0 ;
   - le pipeline de données du CNN personnalisé est inchangé ;
   - 389 862 paramètres et champ réceptif de 46 px ;
   - les tests existants (49) restent verts.
2. **Relance complète** de `custom_cnn` avec la graine 42 dans un **nouveau** dossier
   (`seed_42_nonreg`), comparée à `seed_42` par `scripts/compare_runs.py` : historique,
   probabilités de validation et poids doivent être **identiques au bit près**. Cette relance
   (≈ 5 min de GPU) aura lieu à l'étape 3, avec accord explicite.

## 15. Budget et ressources

- **Temps GPU** (pire cas, 60 époques sans arrêt anticipé, augmentations CPU comprises) :
  - E1 ≈ 1 h, E2 ≈ 1 h, E3 ≈ 4 h ;
  - soit **≈ 6 h au maximum**, probablement 4 à 5 h. Sur secteur.
- **Mémoire GPU** : moins de 2 Go par run, pour 8 Go disponibles.
- **Disque** : ≈ 1,5 Go de checkpoints, non versionnés.
- **Téléchargement** : `resnet50-0676ba61.pth` (97,8 Mo) et `mobilenet_v2-b0353104.pth`
  (13,6 Mo), vérifiés par hash par torchvision. Autorisé **seulement après validation de
  l'implémentation et passage des tests**.
- **Dépendances** : aucune nouvelle (torchvision, scipy et `torch.utils.flop_counter` sont déjà
  disponibles).
- **Licence** : les poids ImageNet sont réservés à un usage non commercial (conditions
  d'ImageNet) ; c'est compatible avec ce projet académique.

## 16. Cohérence avec les phases 1 et 2

- Split, normalisation de la baseline, augmentations, graines, critère de checkpoint et verrou du
  test : **repris à l'identique**.
- Les fichiers, résultats et checkpoints des phases 1 et 2 ne sont **ni modifiés ni réécrits**.
- **Deux précisions**, consignées ici sans modifier les documents d'origine :
  1. `docs/phase2.md` et la docstring de `custom_cnn.py` indiquent que le backward
     d'`AdaptiveAvgPool2d` n'est pas déterministe sur CUDA. C'est exact pour une sortie
     différente de 1×1, mais **pas pour une sortie 1×1** (§13.2). Le choix `x.mean((2,3))` de la
     phase 2 est mathématiquement et numériquement équivalent : **aucun impact sur les
     résultats**.
  2. `docs/phase2.md` (§9, point 4) cite « environ 25 M de paramètres » pour ResNet50. C'est la
     valeur avec la tête à 1 000 classes ; avec la tête à 6 classes, ResNet50 compte
     **23 520 326** paramètres.
- La phase 2 prévoyait de mesurer la latence avec un protocole contrôlé (préchauffage,
  répétitions, médiane) : c'est le protocole du §9.5.

## 17. Livrables prévus

- Code :
  - `ivai/models/pretrained.py` et un registre de modèles ;
  - `ivai/data/subsets.py` ;
  - `ivai/evaluation/calibration.py`, `stats.py` et `benchmark.py` ;
  - adaptations rétrocompatibles du trainer et de `train.py`.
- Configurations du §6 et scripts `benchmark_models.py` et `compare_models.py`.
- Tests :
  - modèles préentraînés ;
  - gel ;
  - groupes de paramètres ;
  - déterminisme ;
  - normalisation ;
  - sous-ensembles et époques virtuelles ;
  - métriques ;
  - benchmark ;
  - non-régression.
- Résultats versionnés : `run.json`, `history.csv`, `val_predictions.csv`, `train_subset.csv`,
  tableaux de synthèse et figures (courbes d'apprentissage, diagrammes de fiabilité, graphique de
  Pareto performance/coût).
- `docs/phase3.md` : déroulé, résultats, écarts au protocole, problèmes rencontrés.

## 18. Hors périmètre de la phase 3

- Toute évaluation sur le test.
- Grad-CAM (phase 6). On note seulement que les cartes finales de ResNet50 et MobileNetV2 font
  7×7 à 200 px, contre 12×12 pour le CNN personnalisé.
- Recalibration des probabilités (par exemple par température). Elle exigerait un ensemble
  distinct de la validation.
- Localisation avec les boîtes de NEU-DET (tâche séparée).
- Backend et frontend.
