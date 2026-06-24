"""Modèle GLOBAL poolé — logique de PRODUCTION (entraînement + features de prédiction).

Un seul XGBoost multi-sortie est entraîné sur TOUTES les stations empilées, avec
la cible et les lags/rolls normalisés par le niveau de chaque station. Le modèle
apprend la *forme* partagée (jour de semaine, saison, météo, vacances) et on
remet à l'échelle par station à la prédiction.

Ce module est volontairement INDÉPENDANT du harnais d'expérimentation
(`evaluation.py`, qui l'importe au contraire pour ses ablations). Il fournit :

  - `build_pooled_matrix` : empile + normalise les stations en un seul (X, Y) ;
  - `train_global`        : entraîne le modèle de prod sur toutes les données et
                            renvoie le bundle à persister (modèle + niveaux +
                            profils + ordre des features).
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from naviflow.config import TARGET
from naviflow.ml_logic.feature_engineering import build_station_profiles
from naviflow.ml_logic.preprocess_xgb import prepare_xgb

# Features d'identité de station (substitut « doux » à un one-hot des 708 stations).
PROFILE_COLS = ["log_vald", "cv", "ratio_we_sem", "ratio_vac_horsvac", "creux_estival"]


def _apply_exclude(df, exclude_window):
    """Retire une fenêtre de dates (ex. COVID). Sans effet si exclude_window est None."""
    if exclude_window is None:
        return df
    start, end = pd.Timestamp(exclude_window[0]), pd.Timestamp(exclude_window[1])
    jour = pd.to_datetime(df["JOUR"])
    return df[(jour < start) | (jour >= end)]


def station_levels_profiles(df, level_cutoff, add_profiles=True):
    """Niveau (moyenne de la cible) et profils par station, calculés sur le TRAIN.

    Seuls les jours STRICTEMENT antérieurs à `level_cutoff` sont utilisés, pour
    ne pas faire fuiter d'info future dans la normalisation.
    """
    train_df = df[pd.to_datetime(df["JOUR"]) < level_cutoff]
    levels = train_df.groupby("ID_LIEU")[TARGET].mean()
    profiles = (build_station_profiles(train_df).set_index("ID_LIEU")[PROFILE_COLS]
                if add_profiles else None)
    return levels, profiles


def build_pooled_matrix(df, station_ids, level_cutoff, horizon=7, lags=(1, 7, 30),
                        rolls=(), normalize=True, add_profiles=True,
                        exclude_window=None):
    """Empile les lignes de toutes les stations en UN jeu (X, Y) poolé et normalisé.

    `level_cutoff` : borne de calcul des niveaux/profils (cf. station_levels_profiles).
    Pour un modèle de prod entraîné sur tout l'historique, passer une date
    postérieure au dernier jour.

    `exclude_window` : (debut, fin) optionnel — fenêtre retirée en amont, pour que
    niveaux, profils ET lags la voient.

    Renvoie (X, Y_abs, gids, dates, feature_names, lvl) :
      X         : features poolées, normalisées (lags/rolls ÷ niveau) si demandé.
      Y_abs     : cible EN ABSOLU (non normalisée).
      gids      : id station de chaque ligne.
      dates     : JOUR de chaque ligne.
      feature_names : noms des colonnes de X, dans l'ordre.
      lvl       : niveau station de chaque ligne (pour (dé)normaliser la cible).
    """
    df = _apply_exclude(df, exclude_window)
    levels, profiles = station_levels_profiles(df, level_cutoff, add_profiles)

    # Pré-groupage : une seule passe au lieu d'un scan complet par station.
    groups = dict(tuple(df.groupby("ID_LIEU", sort=False)))

    X_parts, Y_parts, gid_parts, date_parts, lvl_parts = [], [], [], [], []
    feature_names = None
    min_hist = max(list(lags) + list(rolls) + [1])

    for gid in station_ids:
        if gid not in levels.index or levels[gid] <= 0:
            continue
        if add_profiles and gid not in profiles.index:
            continue
        df_group = groups.get(gid)
        if df_group is None or len(df_group) <= min_hist + horizon + 1:
            continue

        X_np, Y_np, names, dates_np = prepare_xgb(
            df_group, lags=lags, rolls=rolls, horizon=horizon, as_numpy=True)

        if add_profiles:
            prof_block = np.tile(profiles.loc[gid].to_numpy(dtype=float), (len(X_np), 1))
            X_np = np.hstack([X_np, prof_block])
            if feature_names is None:
                feature_names = list(names) + PROFILE_COLS
        elif feature_names is None:
            feature_names = list(names)

        X_parts.append(X_np)
        Y_parts.append(Y_np)
        gid_parts.append(np.full(len(X_np), gid))
        date_parts.append(dates_np)
        lvl_parts.append(np.full(len(X_np), levels[gid]))

    X = np.vstack(X_parts).astype(float)
    Y_abs = np.vstack(Y_parts).astype(float)
    gids = np.concatenate(gid_parts)
    dates = np.concatenate(date_parts).astype("datetime64[ns]")
    lvl = np.concatenate(lvl_parts)

    if normalize:
        lag_idx = [i for i, n in enumerate(feature_names)
                   if n.startswith("lag_") or n.startswith("roll_")]
        if lag_idx:
            X[:, lag_idx] = X[:, lag_idx] / lvl[:, None]

    return X, Y_abs, gids, dates, feature_names, lvl


def train_global(df, station_ids, params, horizon=7, lags=(1, 2, 3, 4, 5, 6, 7),
                 rolls=(7, 14, 30), exclude_window=None, test_cutoff_date=None,
                 random_state=67, multi_strategy="multi_output_tree"):
    """Entraîne le modèle global de PROD et renvoie le bundle à persister.

    test_cutoff_date :
      - None  → déploiement « live » : entraînement sur TOUT l'historique (on
                prédira le vrai futur, jamais vu).
      - date  → déploiement « démo » : entraînement sur les données AVANT le
                cutoff seulement, le reste étant réservé à la démo (dates jamais
                vues). Niveaux/profils sont alors calculés sur le train uniquement
                (pas de fuite du held-out).

    Renvoie le BUNDLE à persister :
      model         : XGBRegressor entraîné (prédit un ratio ~1 par horizon) ;
      levels        : Series {ID_LIEU: niveau} pour (dé)normaliser ;
      profiles      : DataFrame des profils station (features d'identité) ;
      feature_names : ordre EXACT des colonnes attendu à la prédiction ;
      meta          : {lags, rolls, horizon, test_cutoff_date}.
    """
    last_day = pd.to_datetime(df["JOUR"]).max()
    # Cutoff des niveaux/profils : le cutoff de test si holdout (train only), sinon
    # après le dernier jour (tout l'historique).
    level_cutoff = (pd.Timestamp(test_cutoff_date) if test_cutoff_date
                    else last_day + pd.Timedelta(days=1))

    X, Y_abs, _, dates, feature_names, lvl = build_pooled_matrix(
        df, station_ids, level_cutoff=level_cutoff, horizon=horizon, lags=lags,
        rolls=rolls, normalize=True, add_profiles=True, exclude_window=exclude_window)
    Y_fit = Y_abs / lvl[:, None]

    # Holdout : on n'entraîne QUE sur les lignes antérieures au cutoff.
    if test_cutoff_date is not None:
        train_mask = dates < np.datetime64(pd.Timestamp(test_cutoff_date))
        X, Y_fit = X[train_mask], Y_fit[train_mask]

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        multi_strategy=multi_strategy,
        eval_metric="mae",
        random_state=random_state,
        **params,
    )
    model.fit(X, Y_fit)

    # Niveaux/profils à persister : mêmes données (exclusion + cutoff) que l'entraînement.
    df_x = _apply_exclude(df, exclude_window)
    levels, profiles = station_levels_profiles(df_x, level_cutoff, add_profiles=True)
    meta = {"lags": list(lags), "rolls": list(rolls), "horizon": horizon,
            "test_cutoff_date": str(test_cutoff_date) if test_cutoff_date else None}
    return {"model": model, "levels": levels, "profiles": profiles,
            "feature_names": feature_names, "meta": meta}
