"""Sauvegarde et chargement des modeles XGBoost (format natif) + metriques.

Organisation par RUN : chaque entrainement (combinaison grain / horizon /
periode de depart) a son propre sous-dossier, qui contient tous ses modeles
ET son results.csv.

    models_store/
    +-- station_j7_20220701/
    |   +-- xgb_-1006.json
    |   +-- xgb_-1005.json
    |   +-- ...
    |   +-- results.csv
    +-- station_j0_all/
    |   +-- ...
    +-- cluster_j0_all/
        +-- xgb_0.json
        +-- ...
        +-- results.csv

Modeles au format natif XGBoost (.json) : portable, leger, sans risque
d'execution de code au chargement.
"""
import io
import joblib

import pandas as pd
from xgboost import XGBRegressor
from google.cloud import storage
from pathlib import Path
import tempfile


from naviflow.config import (
    PROJECT_ROOT,
    GCP_PROJECT,
    BUCKET_NAME,
    MODEL_TARGET
)

MODELS_STORE = PROJECT_ROOT / "models_store"


def _horizon_tag(horizon):
    """Etiquette horizon : None -> 'j0', N -> 'jN'."""
    return "j0" if horizon is None else f"j{horizon}"


def run_dir(grain="station", horizon=None, train_from=None, suffix=None):
    """Dossier du run pour une combinaison grain / horizon / periode.

    Nom : {grain}_{horizon}_{train_from}  (ex. station_j7_20220701, cluster_j0_all).
    """
    h = _horizon_tag(horizon)
    tf = (train_from or "all").replace("-", "")
    name = f"{grain}_{h}_{tf}"
    if suffix:
        name += f"_{suffix}"
    return MODELS_STORE / name

def _blob_name(group_id, grain="station", horizon=None, train_from=None):
    """Chemin GCS (relatif) du modele, miroir de l'arbo locale."""
    rel = run_dir(grain, horizon, train_from).relative_to(PROJECT_ROOT)
    return f"{rel.as_posix()}/xgb_{group_id}.json"

def model_path(group_id, grain="station", horizon=None, train_from=None):
    """Chemin du fichier modele d'un groupe, dans le sous-dossier de son run."""
    return run_dir(grain, horizon, train_from) / f"xgb_{group_id}.json"


def results_path(grain="station", horizon=None, train_from=None, suffix=None):
    """Chemin du results.csv du run."""
    return run_dir(grain, horizon, train_from, suffix) / "results.csv"


def save_model(model, group_id, grain="station", horizon=None, train_from=None):
    """Sauvegarde un XGBRegressor au format natif, dans le dossier du run."""
    if MODEL_TARGET == "gcs":

        # Format natif XGBoost -> bytes
        model_bytes = bytearray(model.get_booster().save_raw())

        storage_client = storage.Client(project=GCP_PROJECT)
        bucket = storage_client.bucket(BUCKET_NAME)
        destination_blob_name = _blob_name(group_id, grain, horizon, train_from)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_string(bytes(model_bytes), content_type="application/json")
        return destination_blob_name
    else:
        d = run_dir(grain, horizon, train_from)
        d.mkdir(parents=True, exist_ok=True)
        path = model_path(group_id, grain, horizon, train_from)
        model.save_model(path)
        return path


def load_model(group_id, grain="station", horizon=None, train_from=None):
    """Recharge un XGBRegressor, depuis le local ou GCS selon MODEL_TARGET."""
    model = XGBRegressor()
    if MODEL_TARGET == "gcs":
        from google.cloud import storage

        storage_client = storage.Client(project=GCP_PROJECT)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(_blob_name(group_id, grain, horizon, train_from))
        if not blob.exists():
            raise FileNotFoundError(f"Aucun modele sur GCS : {blob.name}")
        model.load_model(bytearray(blob.download_as_bytes()))
    else:
        path = model_path(group_id, grain, horizon, train_from)
        if not path.exists():
            raise FileNotFoundError(f"Aucun modele : {path}")
        model.load_model(path)
    return model


def load_all_models(version: str = "20220101") -> dict:
    models = {}
    client = storage.Client()
    bucket = client.bucket("naviflow-pro-mldl")
    prefix = f"models_store/station_j7_{version}/"

    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".json"):
            continue

        station_id = int(Path(blob.name).stem[4:])

        model = XGBRegressor()
        model.load_model(bytearray(blob.download_as_bytes()))
        models[station_id] = model

    print(f"✅ {len(models)} models loaded from GCS (version {version})")
    return models

def load_all_features() -> dict:
    features = {}

    client = storage.Client()
    bucket = client.bucket("naviflow-pro-mldl")
    prefix = "features_store/"

    blobs = list(bucket.list_blobs(prefix=prefix))

    for blob in blobs:
        if not blob.name.endswith(".parquet"):
            continue

        station_id = int(Path(blob.name).stem[7:])      # retire "X_test_" → "-1001" → -1001

        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        features[station_id] = pd.read_parquet(buffer)

    print(f"✅ {len(features)} feature sets loaded from GCS")
    return features


def save_results(results, grain="station", horizon=None, train_from=None, suffix=None):
    """Ecrit le results.csv dans le sous-dossier du run."""
    d = run_dir(grain, horizon, train_from, suffix)
    d.mkdir(parents=True, exist_ok=True)

    if isinstance(results, dict):
        rows = [{"group_id": gid, **metrics} for gid, metrics in results.items()]
    else:
        rows = list(results)

    df = pd.DataFrame(rows)
    df.insert(0, "grain", grain)
    df.insert(1, "horizon", _horizon_tag(horizon))
    df.insert(2, "train_from", train_from or "all")

    path = results_path(grain, horizon, train_from,suffix)
    df.to_csv(path, index=False)
    return path
