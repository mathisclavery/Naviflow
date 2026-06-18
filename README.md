# Naviflow

Prédiction de l'affluence journalière dans les stations de métro et RER parisiens — jusqu'à 7 jours à l'avance.

---

## Contexte

Naviflow est le projet de fin de bootcamp réalisé avec [Le Wagon](https://www.lewagon.com/) en 2 semaines, en équipe de 4. L'idée de départ : les données de validation des transports parisiens sont publiques depuis 2015, et personne ne s'en sert vraiment pour de la prédiction opérationnelle. On voulait voir jusqu'où on pouvait aller en deux semaines.

L'objectif est de prédire le nombre de validations journalières pour chacune des **708 stations de métro et RER d'Île-de-France**, avec un horizon de prédiction de J+1 à J+7.

---

## Comment ça marche

Le pipeline se décompose en 3 grandes étapes :

**1. Données**
On agrège trois sources : les validations IDFM (2015–2025), la météo journalière de la station Paris-Montsouris (pluie, températures min/max/moy, vent) et un calendrier enrichi (jours fériés, vacances scolaires zone C, ponts).

**2. Feature engineering**
Chaque ligne du dataset correspond à un couple (jour, station). Les features incluent des lags temporels (J-1, J-7, J-30), des encodages cycliques du calendrier (mois, jour de semaine), les variables météo brutes et des features binaires dérivées (pluie significative, vent fort, etc.). Pour le grain "cluster", les stations sont regroupées en 4 clusters KMeans selon leurs profils de trafic.

**3. Modèles**
Deux approches en parallèle :
- **XGBoost** (modèle principal) — un `MultiOutputRegressor` par station ou par cluster, optimisé via `RandomizedSearchCV`. Prédit J+1 à J+7 en une seule passe.
- **RNN encoder-decoder** (modèle alternatif) — prend 140 jours d'historique en entrée et prédit 7 jours. Entraîné sur toutes les stations simultanément.

Les modèles sont servis via une **API FastAPI** déployée sur Cloud Run, et consommés par un frontend **Streamlit**.

---

## Stack technique

 Modélisation : XGBoost, Keras / TensorFlow
 API : FastAPI, Uvicorn
 Frontend : Streamlit
 Infrastructure : Google Cloud Storage, Cloud Run, Docker
 Data : IDFM open data, Météo-France, `holidays` (Python)

---

## Sources de données

- **Validations IDFM** : nombre de passages par station et par jour, de 2015 à 2025. Fichiers CSV/TXT par semestre ou trimestre selon les années — le format a changé plusieurs fois, d'où une couche de détection automatique dans le pipeline.
- **Météo Paris-Montsouris** : données journalières Météo-France (station 75114001). Variables : RR (pluie), TN/TX/TM (températures), FFM (vent moyen).
- **Calendrier** : jours fériés français, vacances scolaires zone C, week-ends, ponts détectés automatiquement.

---

## Installation

```bash
git clone https://github.com/mathisclavery/Naviflow.git
cd Naviflow
pip install -r requirements.txt
pip install -e .
```

Créer un fichier `.env` à la racine (voir `.env.sample`) :

```
GCP_PROJECT=...
GCP_REGION=europe-west1
BUCKET_NAME=...
MODEL_TARGET=local   # ou gcs
```

Configurer les credentials GCP :

```bash
gcloud auth application-default login
```

---

## Utilisation

### Télécharger les données

```bash
make download
```

### Entraîner XGBoost

```bash
make train_xgb                          # par station, tout l'historique
make train_xgb GRAIN=cluster            # par cluster (plus rapide)
make train_xgb TRAIN_FROM=2024-01-01   # test rapide sur données récentes
make train_xgb FORCE=1                  # réentraîner même si les modèles existent
```

### Entraîner le RNN

```bash
python -m naviflow.interface.main_rnn
```

### Lancer l'API

```bash
uvicorn naviflow.api.api:app --reload
```

Endpoints disponibles :

- `GET /ping` — health check
- `GET /predict?station_id=<int>&prediction_date=<YYYY-MM-DD>` — retourne les prédictions J+1 à J+7

---

## Structure du projet

```
Naviflow/
├── naviflow/
│   ├── api/                  # FastAPI
│   ├── config.py             # toute la config en un seul endroit
│   ├── gcp/                  # upload / download GCS
│   ├── interface/            # points d'entrée CLI
│   ├── ml_logic/
│   │   ├── data.py           # chargement et jointure des 3 sources
│   │   ├── feature_engineering.py
│   │   ├── preprocess_xgb.py
│   │   ├── preprocess_rnn.py
│   │   ├── models/           # xgboost, rnn, baselines
│   │   └── sources/          # loaders par source (validations, meteo, calendrier)
│   ├── registry_xgb.py
│   └── registry_rnn.py
├── models_store/             # modèles XGBoost en local
├── rnn/models_store/         # modèles RNN en local
├── notebooks/                # exploration et diagnostics
├── Dockerfile
└── Makefile
```

---

## Déploiement

L'API est containerisée via Docker et déployée sur Google Cloud Run. Les modèles et les données brutes ne sont pas inclus dans l'image — ils sont chargés depuis GCS au démarrage.

```bash
docker build -t naviflow .
docker run -e PORT=8080 -e MODEL_TARGET=gcs -e GCP_PROJECT=... -e BUCKET_NAME=... naviflow
```
