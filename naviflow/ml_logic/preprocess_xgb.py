"""Preprocessing SPECIFIQUE a l'approche tabulaire (XGBoost, regression).

Part du DataFrame enrichi commun (feature_engineering.build_features) et :
  1. ajoute les lags (specifiques tabulaire : le RNN gere le passe autrement) ;
  2. ajoute la target multi-horizon (vecteur J+1 ... J+horizon) ;
  3. selectionne les colonnes de features ;
  4. separe X / Y ;
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
    """Cree une target par horizon : target_j1, target_j2, ..., target_j{horizon}.

    Le modele apprend 'features d'aujourd'hui -> vecteur d'affluence J+1..J+H'.
    Les NaN de fin de serie (pas de futur connu pour le plus grand horizon)
    sont supprimes sur l'ensemble des targets pour garder X et Y alignes.
    """
    df = df.copy()
    df[date] = pd.to_datetime(df[date])
    df = df.sort_values([group, date])

    target_cols = []
    for h in range(1, horizon + 1):
        col = f"target_j{h}"
        df[col] = df.groupby(group)[target].shift(-h)
        target_cols.append(col)

    return df.dropna(subset=target_cols)


# --------------------------------------------------------------------------- #
# 2. Mise en forme X / Y
# --------------------------------------------------------------------------- #


def prepare_xgb(df, lags=(1, 7, 30), horizon=7, keep_id=False,
                onehot_cluster=False, scale=False, as_numpy=False):
    """Prépare X (2D) et Y (2D : une colonne par horizon J+1 ... J+horizon).

    Pensé pour un XGBRegressor multi-sortie (multi_strategy='multi_output_tree')
    qui predit le vecteur [J+1, ..., J+horizon] en une fois.

    Paramètres
    ----------
    df : DataFrame enrichi issu de feature_engineering.build_features().

    lags : décalages a créer. () ou None pour ne pas creer de lags.

    horizon : nombre de jours a predire. Cree les targets target_j1..target_j{horizon}.
              Y aura donc `horizon` colonnes.

    keep_id : si True, conserve ID_LIEU comme feature. A EVITER en general :
              ID_LIEU est un identifiant arbitraire, l'arbre lui donnerait un
              sens d'ordre fictif. Preferer onehot_cluster pour un modele global.

    onehot_cluster : si True, one-hot encode la colonne 'cluster' (4 colonnes
              cluster au lieu d'une feature numerique).

    scale : si True, applique un RobustScaler aux colonnes continues
            (utile pour la regression lineaire ; inutile pour XGBoost).

    as_numpy : si True, renvoie (X_np, Y_np, feature_names) au lieu de
            (X_df, Y_df). XGBoost convertit moins de memoire a partir de
            numpy ; feature_names est conserve a part pour l'importance.

    Renvoie
    -------
    (X, Y) si as_numpy=False          -> X DataFrame, Y DataFrame (colonnes target_jN).
    (X_np, Y_np, feature_names) sinon  -> arrays numpy + liste des noms.
    """

    df = df.copy()

    if lags:
        df = add_lags(df, lags=lags)

    df = add_target_horizon(df, horizon=horizon)
    target_cols = [f"target_j{h}" for h in range(1, horizon + 1)]

    # One-hot du cluster si demande (avant la selection de colonnes)
    if onehot_cluster and "cluster" in df.columns:
        dummies = pd.get_dummies(df["cluster"].astype("Int64"),
                                 prefix="cluster", dtype=int)
        df = pd.concat([df.drop(columns=["cluster"]), dummies], axis=1)

    dates = pd.to_datetime(df["JOUR"]).reset_index(drop=True)

    Y = df[target_cols].copy()

    drop_cols = set(NON_FEATURE_COLS)
    drop_cols.update(target_cols)
    if not keep_id:
        drop_cols.add("ID_LIEU")

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    if scale:
        cols = [c for c in CONTINUOUS_COLS if c in X.columns]
        X[cols] = RobustScaler().fit_transform(X[cols])

    if as_numpy:
        return X.to_numpy(), Y.to_numpy(), list(X.columns), dates.to_numpy()

    return X, Y, dates
