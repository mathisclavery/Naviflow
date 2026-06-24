"""Entraînement du modèle GLOBAL poolé de PRODUCTION.

Entraîne UN seul XGBoost sur toutes les stations empilées (cible normalisée par
niveau station), sur tout l'historique depuis TRAIN_FROM (COVID exclu via
EXCLUDE_WINDOW), avec les hyperparamètres figés (frozen_params_global.json), puis
persiste le bundle (modèle + niveaux + profils + meta) via registry_xgb.

Remplace l'entraînement par station (`main_xgb.py`). Compte ~30 min sur 8 cœurs.

Usage :
    make train_xgb_global
    python -m naviflow.interface.main_xgb_global
"""

import os

from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.global_model import train_global
from naviflow.ml_logic.evaluation import load_frozen_global_params
from naviflow import registry_xgb
from naviflow.utils import display as d
from naviflow.config import TRAIN_FROM, EXCLUDE_WINDOW, DEPLOY_TEST_CUTOFF


def train_and_save(train_from=TRAIN_FROM, exclude_window=EXCLUDE_WINDOW,
                   test_cutoff_date=DEPLOY_TEST_CUTOFF,
                   lags=(1, 2, 3, 4, 5, 6, 7), rolls=(7, 14, 30), horizon=7,
                   save=True):
    """Entraîne le modèle global de prod sur toutes les stations et le persiste."""
    params = load_frozen_global_params()
    params["n_jobs"] = int(os.getenv("N_JOBS", os.cpu_count() or 1))

    d.title("ENTRAÎNEMENT MODÈLE GLOBAL — PROD")
    d.info(f"depuis {train_from} | COVID exclu {exclude_window} | "
           f"holdout démo à partir de {test_cutoff_date} | lags={lags} | rolls={rolls}")

    d.step("Chargement des données + feature engineering")
    df = get_data(train_from=train_from)
    df = build_features(df, with_cluster=False)
    station_ids = sorted(int(s) for s in df["ID_LIEU"].unique())
    d.info(f"{len(df):,} lignes | {len(station_ids)} stations")

    d.step("Entraînement du modèle global poolé (un seul fit)")
    bundle = train_global(df, station_ids, params, horizon=horizon, lags=lags,
                          rolls=rolls, exclude_window=exclude_window,
                          test_cutoff_date=test_cutoff_date)
    d.success(f"Modèle entraîné | {len(bundle['feature_names'])} features | "
              f"{len(bundle['levels'])} niveaux stations")

    if save:
        path = registry_xgb.save_global_bundle(bundle, train_from=train_from)
        d.success(f"Bundle sauvegardé : {path}")

    d.done("Entraînement terminé")
    return bundle


if __name__ == "__main__":
    train_and_save()
