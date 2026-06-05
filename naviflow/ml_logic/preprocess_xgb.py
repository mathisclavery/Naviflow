"""Preprocessing SPECIFIQUE a l'approche tabulaire (XGBoost, regression).

Part du DataFrame enrichi commun (feature_engineering.build_features) et :
  1. ajoute les lags (specifiques tabulaire : le RNN gere le passe autrement) ;
  2. ajoute optionnellement la target horizon (prediction J+N) ;
  3. selectionne les colonnes de features ;
  4. separe X / y ;
  5. scale optionnellement (utile pour la regression lineaire, inutile pour XGBoost).

Toutes les stations sont traitees ensemble (modele global tabulaire).
"""

import pandas as pd
from sklearn.preprocessing import RobustScaler

from naviflow.config import TARGET


# Colonnes a NE jamais passer comme features (identifiants, cibles, libelles)
NON_FEATURE_COLS = ["JOUR", "ID_LIEU", "LIBELLE_ARRET", TARGET]

# Features continues candidates au scaling (pour la regression lineaire)
CONTINUOUS_COLS = ["RR", "TN", "TX", "TM", "FFM", "amplitude", "log_RR",
                   "lag_1", "lag_7", "lag_30"]


# --------------------------------------------------------------------------- #
# 1. Lags (specifiques tabulaire)
# --------------------------------------------------------------------------- #
def add_lags(df, lags=(1, 7, 30), target=TARGET, group="ID_LIEU", date="JOUR"):
    """Ajoute lag_1, lag_7, lag_30 calcules PAR STATION (groupby + shift).

    Le groupby(station) est crucial : sans lui, le lag de la premiere ligne
    d'une station emprunterait a la station precedente. Les NaN de debut de
    serie sont supprimes.
    """
    df = df.copy()
    df[date] = pd.to_datetime(df[date])
    df = df.sort_values([group, date])

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group)[target].shift(lag)

    return df.dropna(subset=[f"lag_{lag}" for lag in lags])


def add_target_horizon(df, horizon=7, target=TARGET, group="ID_LIEU", date="JOUR"):
    """Cree une target decalee de `horizon` jours dans le futur (target_jN).

    Le modele apprend 'features d'aujourd'hui -> affluence dans N jours'.
    Les NaN de fin de serie (pas de futur connu) sont supprimes.
    """
    df = df.copy()
    df[date] = pd.to_datetime(df[date])
    df = df.sort_values([group, date])

    target_col = f"target_j{horizon}"
    df[target_col] = df.groupby(group)[target].shift(-horizon)

    return df.dropna(subset=[target_col])


# --------------------------------------------------------------------------- #
# 2. Mise en forme X / y
# --------------------------------------------------------------------------- #


def prepare_xgb(df, lags=(1, 7, 30), horizon=None, keep_id=False,
                onehot_cluster=False, scale=False, as_numpy=False):
    """Prépare X (2D) et y pour un modele tabulaire.

    Paramètres
    ----------
    df : DataFrame enrichi issu de feature_engineering.build_features().

    lags : décalages a créer. () ou None pour ne pas creer de lags.

    horizon : si renseigné (ex. 7), créé target_j{horizon} et l'utilise comme y.
              Si None, y = TARGET (prediction du jour courant).

    keep_id : si True, conserve ID_LIEU comme feature. A EVITER en general :
              ID_LIEU est un identifiant arbitraire, l'arbre lui donnerait un
              sens d'ordre fictif. Preferer onehot_cluster pour un modele global.

    onehot_cluster : si True, one-hot encode la colonne 'cluster' (4 colonnes
              cluster au lieu d'une feature numerique).

    scale : si True, applique un RobustScaler aux colonnes continues
            (utile pour la regression lineaire ; inutile pour XGBoost).

    as_numpy : si True, renvoie (X_np, y_np, feature_names) au lieu de
            (X_df, y_series). XGBoost convertit moins de memoire a partir de
            numpy ; feature_names est conserve a part pour l'importance.

    Renvoie
    -------
    (X, y) si as_numpy=False         -> X DataFrame, y Series.
    (X_np, y_np, feature_names) sinon -> arrays numpy + liste des noms.
    """

    df = df.copy()

    if lags:
        df = add_lags(df, lags=lags)

    if horizon is not None:
        df = add_target_horizon(df, horizon=horizon)
        target_col = f"target_j{horizon}"
    else:
        target_col = TARGET

    # One-hot du cluster si demande (avant la selection de colonnes)
    if onehot_cluster and "cluster" in df.columns:
        dummies = pd.get_dummies(df["cluster"].astype("Int64"),
                                 prefix="cluster", dtype=int)
        df = pd.concat([df.drop(columns=["cluster"]), dummies], axis=1)

    y = df[target_col]

    # Colonnes a exclure des features
    drop_cols = set(NON_FEATURE_COLS)
    drop_cols.add(target_col)
    if not keep_id:
        drop_cols.add("ID_LIEU")

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    if scale:
        cols = [c for c in CONTINUOUS_COLS if c in X.columns]
        X[cols] = RobustScaler().fit_transform(X[cols])

    if as_numpy:
        return X.to_numpy(), y.to_numpy(), list(X.columns)

    return X, y
