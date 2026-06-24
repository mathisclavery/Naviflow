"""Pré-calcul des features de SERVICE du modèle global.

Construit, pour chaque station, ses features déjà normalisées (au format exact
attendu par le modèle global) pour les `DAYS` derniers jours, et les persiste via
registry_xgb. L'API charge ces parquets pour servir une prédiction sans recalculer.

La construction réutilise `build_pooled_matrix` avec EXACTEMENT les mêmes
paramètres que l'entraînement (même fenêtre, même exclusion, même cutoff de
niveau) : la normalisation est donc identique à celle vue par le modèle.

Usage :
    make save_features_global
    DAYS=180 python -m naviflow.interface.main_save_features_global
"""

import os

import numpy as np
import pandas as pd

from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.global_model import build_pooled_matrix
from naviflow import registry_xgb
from naviflow.utils import display as d
from naviflow.config import TRAIN_FROM, EXCLUDE_WINDOW, DEPLOY_TEST_CUTOFF


def save_features(train_from=TRAIN_FROM, exclude_window=EXCLUDE_WINDOW,
                  test_cutoff_date=DEPLOY_TEST_CUTOFF,
                  lags=(1, 2, 3, 4, 5, 6, 7), rolls=(7, 14, 30), horizon=7,
                  days=120, save=True):
    """Pré-calcule et persiste les features de service.

    En mode démo (test_cutoff_date défini), on sert exactement la fenêtre held-out
    (dates ≥ cutoff) : ce sont des jours jamais vus à l'entraînement. La
    normalisation utilise le MÊME cutoff de niveau que l'entraînement (train only),
    donc des features cohérentes. Sinon (live), on sert les `days` derniers jours.
    """
    d.title("FEATURES DE SERVICE — MODÈLE GLOBAL")

    d.step("Chargement des données + feature engineering")
    df = get_data(train_from=train_from)
    df = build_features(df, with_cluster=False)
    station_ids = sorted(int(s) for s in df["ID_LIEU"].unique())

    last_day = pd.to_datetime(df["JOUR"]).max()
    # Cutoff de niveau identique à l'entraînement (train only si holdout).
    level_cutoff = (pd.Timestamp(test_cutoff_date) if test_cutoff_date
                    else last_day + pd.Timedelta(days=1))

    d.step("Construction des features normalisées (build_pooled_matrix)")
    X, _, gids, dates, feature_names, _ = build_pooled_matrix(
        df, station_ids, level_cutoff=level_cutoff, horizon=horizon, lags=lags,
        rolls=rolls, normalize=True, add_profiles=True, exclude_window=exclude_window)

    # Fenêtre servable : le held-out (dates ≥ cutoff) en mode démo, sinon les
    # `days` derniers jours.
    if test_cutoff_date is not None:
        servable = dates >= np.datetime64(pd.Timestamp(test_cutoff_date))
        d.info(f"held-out servable à partir de {test_cutoff_date}")
    else:
        servable = dates >= (last_day - pd.Timedelta(days=days)).to_datetime64()
        d.info(f"{days} derniers jours servables")

    feat = pd.DataFrame(X[servable], columns=feature_names)
    feat["JOUR"] = pd.to_datetime(dates[servable])
    feat["ID_LIEU"] = gids[servable]

    features_by_station = {int(gid): g.drop(columns=["ID_LIEU"])
                           for gid, g in feat.groupby("ID_LIEU")}
    d.info(f"{len(features_by_station)} stations | {len(feat)} lignes servables")

    if save:
        path = registry_xgb.save_global_features(features_by_station,
                                                 train_from=train_from, horizon=horizon)
        d.success(f"Features sauvegardées : {path}")

    d.done("Features de service prêtes")
    return features_by_station


if __name__ == "__main__":
    save_features(days=int(os.getenv("DAYS", "120")))
