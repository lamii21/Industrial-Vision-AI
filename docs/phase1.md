# Phase 1 : environnement, données et exploration

Date d'exécution : 14/11/2026. Toutes les valeurs ci-dessous ont été mesurées réellement.

## 1. Environnement vérifié

| Élément | Version / valeur |
|---|---|
| OS | Windows 11 Pro (10.0.26300) |
| Python (venv) | 3.12.8 |
| GPU | NVIDIA RTX A2000 8 GB Laptop, compute capability 8.6 (Ampere), bf16 supporté |
| Pilote NVIDIA | 610.60 (supporte CUDA jusqu'à 13.3) |
| PyTorch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 (build CUDA 13.0, cuDNN 9.24) |
| numpy / pandas / scipy / scikit-learn | 2.5.4 / 3.1.1 / 1.18.1 / 1.9.1 |
| matplotlib / pillow / opencv-headless | 3.11.2 / 13.0.0 / 5.0.0 |
| Test GPU (matmul 2048x2048 fp32) | GPU 4,84 ms contre CPU 250,8 ms, erreur relative 1,2e-6 |
| Test AMP (conv fp16, batch 32, 224x224) | OK |

**Pourquoi Python 3.12 et pas 3.13 ?** Les deux sont installés. 3.12 est la version la plus
largement supportée par l'écosystème scientifique (wheels précompilées disponibles pour
toutes les dépendances), ce qui limite les risques d'installation.

## 2. Installation (PowerShell, depuis la racine du projet)

```powershell
# 1) Environnement virtuel isolé : les paquets du projet ne touchent pas au Python système
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2) PyTorch avec CUDA : le wheel PyPI standard pour Windows est CPU-only,
#    il faut donc utiliser l'index PyTorch spécifique à CUDA 13.0
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130

# 3) Autres dépendances (versions figées)
pip install -r ai-engine/requirements.txt

# 4) Le code du projet installé comme paquet "éditable" : `import ivai` fonctionne
#    partout (scripts, notebooks, et plus tard le backend) sans bricoler sys.path
pip install -e ai-engine

# 5) Noyau Jupyter lié au venv (sinon Jupyter peut utiliser un autre Python)
python -m ipykernel install --user --name ivai --display-name "Python 3.12 (ivai)"
```

Pour une reproduction exacte de toutes les dépendances transitives :
`pip install -r ai-engine/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130`.

## 3. Exécution de la phase 1 (depuis `ai-engine/`)

| Commande | Rôle | Durée indicative (cette machine) |
|---|---|---|
| `python scripts/check_env.py` | Affiche les versions, vérifie que le GPU **calcule juste** (comparaison CPU/GPU) et que l'AMP fonctionne. Code retour 1 si un problème est détecté. | ~10 s |
| `python scripts/download_data.py` | Télécharge NEU-CLS (Figshare, CC BY 4.0), **vérifie le MD5** publié, extrait de façon sécurisée. | ~10 s |
| `python scripts/prepare_splits.py` | Index des images, détection des doublons, split stratifié par groupes, sauvegarde des CSV, copie en `train/val/test/<classe>/`. | ~2 min 30 |
| `python scripts/verify_splits.py` | 8 contrôles indépendants (fuite, stratification, intégrité). Code retour 1 en cas d'échec. | ~40 s |
| `python -m pytest` | 22 tests unitaires (split, doublons, téléchargement, index). | ~1-2 min |
| `jupyter lab notebooks/01_eda.ipynb` | Exploration interactive (noyau « Python 3.12 (ivai) »). | – |
| `jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb` | Réexécute tout le notebook sans interface (reproductibilité des figures). | ~1 min |

## 4. Choix techniques à savoir défendre

**Source des données.** Le site officiel (Northeastern University) propose NEU-CLS via
Google Drive ou Baidu, ce qui impose un téléchargement manuel. Le miroir Figshare est
scriptable, sous licence CC BY 4.0, et publie un MD5. Ce miroir contient les **mêmes
1 800 images** au format JPEG avec des annotations YOLO (issues de NEU-DET). Son découpage
train/valid (1771/31) est ignoré : il est arbitraire et non stratifié.

**Sécurité de l'extraction.** Seuls les `.jpg` et `.txt` sont extraits. Les chemins
contenant `..` sont refusés (attaque « zip slip »). Les fichiers `labels.cache` sont des
*pickles* Python : les charger pourrait exécuter du code arbitraire, ils sont donc ignorés.

**Étiquette = nom de fichier.** `scratches_12.jpg` → `scratches`. Les boîtes YOLO servent
uniquement de contrôle croisé et pour préparer la future extension de localisation.

**Split stratifié 70/15/15, groupé, figé.**
- *Stratifié* : chaque subset contient exactement 1/6 de chaque classe (210/45/45).
- *Groupé* : des images doublons l'une de l'autre restent dans le **même** subset.
- *Figé* : seed 42 + CSV versionnés avec Git. Vérifié : relancer le script produit des CSV
  identiques au bit près (même SHA-256).
- Le **test** ne sera utilisé qu'une seule fois, pour l'évaluation finale. Les
  hyperparamètres seront choisis sur la **validation**.

**Détection de doublons en deux étapes.**
1. pHash (64 bits issus de la DCT basse fréquence) : deux images visuellement proches ont
   des hash proches. On prend la distance minimale sur les **8 rotations et symétries**, pour
   trouver aussi les copies tournées.
2. Confirmation par corrélation de Pearson des pixels (≥ 0,95), pour éviter les faux
   positifs sur des textures qui se ressemblent.

## 5. Résultats obtenus

- **1 800 images**, 6 classes x 300, 200x200, contenu 100 % niveaux de gris (stocké en RGB).
- **Split** : train 1 260, val 270, test 270 (210/45/45 par classe). **8/8 contrôles passés.**
- **Doublons** : 1 exact (`patches_101` = `patches_105`), 1 quasi-doublon tourné de 90°
  (`scratches_232` ~ `scratches_247`, r = 0,957). Les deux paires sont dans le train.
  2 autres candidats ont été rejetés (r ≤ 0,42). Distance pHash minimale entre subsets
  différents : 6, sans aucun doublon confirmé.
- **Images multi-défauts** : 123 (6,8 %) contiennent un défaut secondaire annoté, la classe
  du nom figurant toujours parmi les boîtes.
- **Normalisation (train uniquement)** : moyenne 0,5043, écart-type 0,2125.
- **Constats de l'EDA** : fortes différences de luminosité et de contraste entre classes
  (risque de raccourci), bord droit plus sombre sur les images moyennes (artefact
  d'acquisition), rayures dans les deux orientations.

Figures : `ai-engine/reports/figures/fig01` à `fig09`.

## 6. Problèmes rencontrés et solutions

| Problème | Cause | Solution appliquée |
|---|---|---|
| Installation de PyTorch extrêmement lente (~100 Ko/s) | Réseau instable, et **deux `pip` lancés en parallèle** (le premier, issu d'une session interrompue, tournait encore) | Processus en double arrêtés ; téléchargement du wheel (1,99 Go) en 8 morceaux parallèles avec reprise, **vérification du SHA-256** publié par PyTorch, puis installation depuis le fichier local |
| `pip` : « No matching distribution found for jupyterlab » | Coupure réseau passagère (PyPI joignable juste après) | Relance avec `--retries 10 --timeout 60` |
| Mesure GPU initiale : 225 ms (aberrante) | Le premier appel CUDA inclut l'initialisation de cuBLAS | Ajout d'un **préchauffage** (warm-up) avant la mesure : 4,84 ms. Même règle pour les futurs benchmarks de latence |
| 123 « incohérences » entre étiquettes YOLO et nom de fichier | Ce ne sont pas des erreurs : images **multi-défauts** | Contrôle redéfini : erreur seulement si la classe du nom est absente des boîtes (0 cas) ; les défauts secondaires sont tracés comme information |
| Détection de doublons lente (5 min 44 s au total) | Distances de Hamming calculées bit par bit | Hash stockés en entiers 64 bits + `XOR` + comptage de bits : 2,2 s pour toutes les paires |
| Notebook : `ModuleNotFoundError: ivai` | Le noyau `python3` pointait vers un autre Python (Anaconda) | Noyau dédié `ivai` enregistré depuis le venv |
| Accents illisibles dans la console | Console Windows en cp1252 | `sys.stdout.reconfigure(encoding="utf-8")` |
| Docker absent | Non installé sur la machine | Sans impact en phase 1. À régler avant la phase 7 : installer Docker Desktop, ou PostgreSQL natif |

## 7. Ce qui n'a PAS été fait (volontairement)

Aucun modèle n'a été entraîné. Aucun code backend ou frontend n'a été écrit. Aucune donnée
synthétique n'a été générée.
