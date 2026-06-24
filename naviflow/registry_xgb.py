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
import json
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


# --------------------------------------------------------------------------- #
# Modele GLOBAL poole : un seul run contenant modele + niveaux + profils + meta
# --------------------------------------------------------------------------- #
# Le modele global n'est PAS un modele par groupe : son run_dir (global_j{H}_...)
# contient un unique bundle. On serialise tout en JSON natif (pas de pickle) :
#   xgb_global.json  booster XGBoost natif
#   levels.json      {station_id: niveau} pour (de)normaliser
#   profiles.json    {station_id: {profil: valeur}} (features d'identite)
#   meta.json        {feature_names, lags, rolls, horizon}
_GLOBAL_FILES = {"model": "xgb_global.json", "levels": "levels.json",
                 "profiles": "profiles.json", "meta": "meta.json"}


def _global_payload(bundle):
    """Convertit le bundle (objets pandas/XGBoost) en dict JSON-serialisables."""
    levels = {int(k): float(v) for k, v in bundle["levels"].items()}
    profiles = {int(k): {c: float(x) for c, x in row.items()}
                for k, row in bundle["profiles"].to_dict(orient="index").items()}
    meta = {**bundle["meta"], "feature_names": list(bundle["feature_names"])}
    return levels, profiles, meta


def save_global_bundle(bundle, train_from=None):
    """Sauvegarde le bundle du modele global (local ou GCS selon MODEL_TARGET)."""
    horizon = bundle["meta"]["horizon"]
    model = bundle["model"]
    levels, profiles, meta = _global_payload(bundle)
    blobs = {"levels": json.dumps(levels), "profiles": json.dumps(profiles),
             "meta": json.dumps(meta)}

    if MODEL_TARGET == "gcs":
        rel = run_dir("global", horizon, train_from).relative_to(PROJECT_ROOT).as_posix()
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(BUCKET_NAME)
        bucket.blob(f"{rel}/{_GLOBAL_FILES['model']}").upload_from_string(
            bytes(bytearray(model.get_booster().save_raw())), content_type="application/json")
        for key, text in blobs.items():
            bucket.blob(f"{rel}/{_GLOBAL_FILES[key]}").upload_from_string(
                text, content_type="application/json")
        return f"gs://{BUCKET_NAME}/{rel}"

    d = run_dir("global", horizon, train_from)
    d.mkdir(parents=True, exist_ok=True)
    model.save_model(d / _GLOBAL_FILES["model"])
    for key, text in blobs.items():
        (d / _GLOBAL_FILES[key]).write_text(text)
    return d


def load_global_bundle(train_from=None, horizon=7):
    """Recharge le bundle du modele global (local ou GCS selon MODEL_TARGET)."""
    model = XGBRegressor()

    if MODEL_TARGET == "gcs":
        rel = run_dir("global", horizon, train_from).relative_to(PROJECT_ROOT).as_posix()
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(BUCKET_NAME)
        model.load_model(bytearray(
            bucket.blob(f"{rel}/{_GLOBAL_FILES['model']}").download_as_bytes()))
        texts = {key: bucket.blob(f"{rel}/{_GLOBAL_FILES[key]}").download_as_text()
                 for key in ("levels", "profiles", "meta")}
    else:
        d = run_dir("global", horizon, train_from)
        model_path_ = d / _GLOBAL_FILES["model"]
        if not model_path_.exists():
            raise FileNotFoundError(f"Aucun modele global : {model_path_}")
        model.load_model(model_path_)
        texts = {key: (d / _GLOBAL_FILES[key]).read_text()
                 for key in ("levels", "profiles", "meta")}

    levels = pd.Series({int(k): v for k, v in json.loads(texts["levels"]).items()})
    profiles = pd.DataFrame.from_dict(json.loads(texts["profiles"]), orient="index")
    profiles.index = profiles.index.astype(int)
    meta = json.loads(texts["meta"])
    return {"model": model, "levels": levels, "profiles": profiles,
            "feature_names": meta["feature_names"], "meta": meta}


# --------------------------------------------------------------------------- #
# Features de SERVICE du modele global (pre-calculees, normalisees)
# --------------------------------------------------------------------------- #
# Un parquet par station (JOUR + colonnes de features deja normalisees), range
# dans le sous-dossier features/ du run global. L'API les charge pour servir une
# date donnee sans recalculer.
def _global_features_dir(horizon=7, train_from=None):
    return run_dir("global", horizon, train_from) / "features"


def save_global_features(features_by_station, train_from=None, horizon=7):
    """Sauvegarde les features de service (dict {station_id: DataFrame})."""
    if MODEL_TARGET == "gcs":
        rel = _global_features_dir(horizon, train_from).relative_to(PROJECT_ROOT).as_posix()
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(BUCKET_NAME)
        for gid, df_feat in features_by_station.items():
            buffer = io.BytesIO()
            df_feat.to_parquet(buffer, index=False)
            buffer.seek(0)
            bucket.blob(f"{rel}/X_{gid}.parquet").upload_from_file(
                buffer, content_type="application/octet-stream")
        return f"gs://{BUCKET_NAME}/{rel}"

    d = _global_features_dir(horizon, train_from)
    d.mkdir(parents=True, exist_ok=True)
    for gid, df_feat in features_by_station.items():
        df_feat.to_parquet(d / f"X_{gid}.parquet", index=False)
    return d


def load_global_features(train_from=None, horizon=7):
    """Recharge les features de service : dict {station_id: DataFrame}."""
    features = {}
    if MODEL_TARGET == "gcs":
        rel = _global_features_dir(horizon, train_from).relative_to(PROJECT_ROOT).as_posix()
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(BUCKET_NAME)
        for blob in bucket.list_blobs(prefix=f"{rel}/"):
            if not blob.name.endswith(".parquet"):
                continue
            gid = int(Path(blob.name).stem[2:])  # "X_<id>" -> id
            buffer = io.BytesIO(blob.download_as_bytes())
            features[gid] = pd.read_parquet(buffer)
    else:
        d = _global_features_dir(horizon, train_from)
        for path in d.glob("X_*.parquet"):
            features[int(path.stem[2:])] = pd.read_parquet(path)
    return features
