# Naviflow — Prédiction de fréquentation du métro parisien

Naviflow prédit le nombre de validations journalières par station de métro (J+1) à partir de données historiques IDFM, météo et calendrier.

---

## Prérequis

- Python 3.10
- Un compte Google Cloud avec accès au projet `naviflow-pro`
- `gcloud` CLI installé ([guide d'installation](https://cloud.google.com/sdk/docs/install))

---

## Installation

```bash
git clone https://github.com/ton-repo/Naviflow.git
cd Naviflow
pip install -r requirements.txt
pip install -e .
```

---

## Configuration des credentials GCP

Les données sont stockées dans un bucket Google Cloud Storage. Avant de lancer quoi que ce soit, configure tes credentials GCP.

### Option A — Compte de service (recommandé si tu as un fichier `.json`)

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/chemin/vers/ton-fichier.json"
```

Pour que ce soit permanent, ajoute cette ligne à ton `~/.bashrc` ou `~/.zshrc` :

```bash
echo 'export GOOGLE_APPLICATION_CREDENTIALS="/chemin/vers/ton-fichier.json"' >> ~/.zshrc
source ~/.zshrc
```

### Option B — Authentification personnelle via gcloud

```bash
gcloud auth application-default login
```

Une fenêtre s'ouvre dans ton navigateur — connecte-toi avec le compte Google associé au projet `naviflow-pro`.

### Vérification

```bash
gcloud storage ls gs://naviflow-pro-mldl/
```

Tu dois voir `gs://naviflow-pro-mldl/raw/` s'afficher.

---

## Utilisation

### 1. Télécharger les données depuis GCS

```bash
make download
```

Télécharge `raw_data/` depuis le bucket GCS vers ton disque local.
Les fichiers déjà présents sont ignorés (reprise possible si interruption).

### 2. Lancer la baseline

```bash
make baseline
```

Entraîne deux modèles naïfs sur la station avec le plus de données :
- **Baseline moyenne** : prédit toujours la moyenne historique
- **Baseline lag J-7** : prédit l'affluence du même jour la semaine précédente

Affiche MAE, R² et erreur relative — ce sont les scores plancher à battre.

### 3. Autres commandes disponibles

```bash
make preprocess   # charge les données + feature engineering
make train        # entraîne XGBoost sur toutes les stations
make evaluate     # affiche le résumé des métriques sauvegardées
make predict      # prédit sur une station avec un modèle sauvegardé
make upload_raw   # upload raw_data/ local vers GCS
```

---

## Structure du projet

```
Naviflow/
├── naviflow/
│   ├── interface/
│   │   └── main.py          # point d'entrée CLI
│   ├── ml_logic/
│   │   ├── data.py           # chargement et jointure des données
│   │   ├── feature_engineering.py
│   │   ├── preprocess_xgb.py
│   │   └── models/
│   │       ├── baselines.py  # modèles naïfs
│   │       └── sklearn_models.py  # XGBoost
│   ├── gcp/
│   │   ├── gcs_loader.py     # téléchargement depuis GCS
│   │   └── upload_raw_data.py
│   ├── config.py
│   └── params.py
├── raw_data/                 # données locales (téléchargées via make download)
├── models_store/             # modèles sauvegardés après make train
├── requirements.txt
├── Makefile
└── setup.py
```

---

## Sources de données

| Source | Description |
|--------|-------------|
| IDFM validations | Nombre de validations par station et par jour (2015-2025) |
| Météo-France | Données journalières Paris-Montsouris (RR, TN, TX, TM, FFM) |
| Calendrier | Jours fériés, vacances scolaires, week-ends (zone C) |
