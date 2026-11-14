# Industrial Vision AI – Smart Quality Inspection

Plateforme de classification automatique des défauts de surface sur des tôles d'acier
(dataset **NEU-CLS**), avec CNN personnalisé, ResNet50 et MobileNetV2, interprétabilité
Grad-CAM et application web (FastAPI + React).

> **Statut : Phase 1 terminée** (environnement, données, split, EDA). Aucun modèle n'est
> encore entraîné. Voir [docs/phase1.md](docs/phase1.md).

## Principe de conformité

Le modèle produit une **prédiction** (classe + score de confiance). Ce n'est **pas** une
décision de conformité : le statut final d'une pièce est attribué ou validé par un
opérateur qualité. L'application distinguera toujours :
`prédiction du modèle` ≠ `statut de conformité validé`.

## Structure

```
projet/
├── ai-engine/                 # moteur Deep Learning (paquet Python « ivai »)
│   ├── ivai/
│   │   ├── config.py          # chemins + chargement de configs/data.yaml
│   │   ├── data/              # téléchargement, index, doublons, split, vérifications
│   │   ├── models/            # (phase 3-4) custom_cnn, resnet50, mobilenetv2
│   │   ├── training/          # (phase 5)
│   │   ├── evaluation/        # (phase 5-6) métriques, comparaison expérimentale
│   │   ├── explainability/    # (phase 6) Grad-CAM
│   │   └── inference/         # (phase 9) prédicteur partagé avec le backend
│   ├── configs/data.yaml      # source, MD5, classes, seed, ratios, seuils doublons
│   ├── scripts/               # check_env, download_data, prepare_splits, verify_splits
│   ├── notebooks/01_eda.ipynb # exploration des données
│   ├── datasets/
│   │   ├── raw/               # (non versionné) archive + images extraites
│   │   ├── processed/         # (non versionné) train/val/test/<classe>/
│   │   └── splits/            # (versionné) CSV du split + index + métadonnées
│   ├── reports/figures/       # figures de l'EDA
│   ├── tests/                 # pytest
│   ├── requirements.txt
│   └── pyproject.toml
├── backend/                   # (phase 7) FastAPI
├── frontend/                  # (phase 8) React + TypeScript
├── artifacts/models/          # (phase 5) poids entraînés (non versionnés)
└── docs/
```

## Démarrage rapide (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r ai-engine/requirements.txt
pip install -e ai-engine

cd ai-engine
python scripts/check_env.py        # versions + test réel sur GPU
python scripts/download_data.py    # télécharge NEU-CLS + vérifie le MD5
python scripts/prepare_splits.py   # index, doublons, split stratifié 70/15/15
python scripts/verify_splits.py    # contrôles anti-fuite (doit afficher 8/8)
python -m pytest                   # tests unitaires
jupyter lab notebooks/01_eda.ipynb
```

Le détail et l'explication de chaque commande sont dans [docs/phase1.md](docs/phase1.md).

## Dataset

NEU surface defect database, Northeastern University (Song K., Yan Y. et al.).
Miroir utilisé : [Figshare – NEU-CLS](https://figshare.com/articles/dataset/NEU-CLS/28903550) (CC BY 4.0).
À citer : Song & Yan, *Applied Surface Science*, 2013 ; Bao et al., *IEEE TIM*, 2021.
