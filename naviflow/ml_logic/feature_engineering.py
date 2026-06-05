"""Feature engineering COMMUN aux deux paradigmes (XGBoost et RNN).

Ce module est la *couche pivot* : il prend le DataFrame brut issu de
`data.get_data()` et lui ajoute des colonnes nommees (temporelles, meteo
derivees, cluster optionnel). Il NE scale RIEN et NE cree PAS de lags.

  - Le scaling est specifique a chaque modele (preprocess_xgb / preprocess_rnn).
  - Les lags et la target horizon sont specifiques a l'approche tabulaire
    (preprocess_xgb), car le RNN gere le passe via ses sequences temporelles.

Sortie garantie : un DataFrame au grain (JOUR, ID_LIEU) enrichi, ou toutes
les features sont des colonnes nommees. C'est le format que consomment
`preprocess_xgb.py` et `preprocess_rnn.py`.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from naviflow.config import N_CLUSTERS_DEFAULT


# --------------------------------------------------------------------------- #
# Seuils meteo (features binaires derivees)
# --------------------------------------------------------------------------- #
SEUIL_PLUIE = 1.0    # mm/j  — pluie significative
SEUIL_VENT  = 12.0   # m/s   — vent fort
SEUIL_FROID = 5.0    # °C    — froid ressenti
SEUIL_CHAUD = 28.0   # °C    — chaleur significative


# --------------------------------------------------------------------------- #
# 1. Features temporelles (ajoutees comme colonnes nommees)
# --------------------------------------------------------------------------- #
def add_time_features(df):
    """Ajoute les features calendaires derivees de JOUR.

    Colonnes ajoutees :
      jour_semaine (0=lundi), jour_mois, jour_annee, semaine_annee, mois,
      trimestre, annee, is_debut_mois, is_fin_mois, mois_sin, mois_cos.

    mois_sin / mois_cos : encodage cyclique du mois. Inutile aux arbres
    (XGBoost) mais utile au RNN (PAST_COVARIATES). Produit pour tout le monde ;
    XGBoost les ignorera ou non selon sa selection de colonnes.
    """
    df = df.copy()
    dates = pd.to_datetime(df["JOUR"])

    df["jour_semaine"]  = dates.dt.dayofweek
    df["jour_mois"]     = dates.dt.day
    df["jour_annee"]    = dates.dt.dayofyear
    df["semaine_annee"] = dates.dt.isocalendar().week.astype(int)
    df["mois"]          = dates.dt.month
    df["trimestre"]     = dates.dt.quarter
    df["annee"]         = dates.dt.year
    df["is_debut_mois"] = (df["jour_mois"] <= 5).astype(int)
    df["is_fin_mois"]   = (df["jour_mois"] >= 25).astype(int)
    df["mois_sin"]      = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"]      = np.cos(2 * np.pi * df["mois"] / 12)

    return df


# --------------------------------------------------------------------------- #
# 2. Features meteo derivees (ajoutees comme colonnes nommees, NON scalees)
# --------------------------------------------------------------------------- #
def add_weather_features(df):
    """Ajoute les features meteo derivees a partir de RR, TN, TX, TM, FFM.

    Colonnes ajoutees :
      amplitude (TX-TN), log_RR, is_rain, is_wind, is_cold, is_hot,
      meteo_degradee.

    Les colonnes brutes RR/TN/TX/TM/FFM sont conservees telles quelles
    (le scaling, si necessaire, est fait en aval par chaque modele).
    L'imputation des NaN meteo est faite par mediane.
    """
    df = df.copy()

    # Imputation mediane des colonnes meteo brutes
    meteo_brutes = ["RR", "TN", "TX", "TM", "FFM"]
    for col in meteo_brutes:
        df[col] = df[col].fillna(df[col].median())

    # Features continues derivees
    df["amplitude"] = df["TX"] - df["TN"]
    df["log_RR"]    = np.log1p(df["RR"])

    # Features binaires derivees
    df["is_rain"] = (df["RR"]  >= SEUIL_PLUIE).astype(int)
    df["is_wind"] = (df["FFM"] >= SEUIL_VENT).astype(int)
    df["is_cold"] = (df["TM"]  <= SEUIL_FROID).astype(int)
    df["is_hot"]  = (df["TM"]  >= SEUIL_CHAUD).astype(int)

    return df


# --------------------------------------------------------------------------- #
# 3. Clustering des stations (optionnel)
# --------------------------------------------------------------------------- #
def build_station_profiles(df):
    """Agrege le dataset journalier en un profil par station.

    Une ligne = une station, decrite par : log_vald, cv, ratio_we_sem,
    ratio_vac_horsvac, creux_estival. Necessite IS_WEEKEND / IS_VACANCES.
    """
    df = df.copy()  # evite l'effet de bord IS_AOUT sur le df appelant

    # Normalise les indicateurs en booleen : robuste que la source les fournisse
    # en bool (calendrier.py) ou en 0/1 (selon les jointures/relectures).
    for col in ["IS_WEEKEND", "IS_VACANCES"]:
        df[col] = df[col].astype(bool)

    profiles = df.groupby("ID_LIEU").agg(
        nb_vald_moyen=("NB_VALD_TOTAL", "mean"),
        nb_vald_std=("NB_VALD_TOTAL", "std"),
    ).reset_index()

    wk = df.groupby(["ID_LIEU", "IS_WEEKEND"])["NB_VALD_TOTAL"].mean().unstack()
    profiles["ratio_we_sem"] = profiles["ID_LIEU"].map(wk[True] / wk[False])

    vac = df.groupby(["ID_LIEU", "IS_VACANCES"])["NB_VALD_TOTAL"].mean().unstack()
    profiles["ratio_vac_horsvac"] = profiles["ID_LIEU"].map(vac[True] / vac[False])

    df["IS_AOUT"] = df["JOUR"].dt.month.isin([8])
    ete = df.groupby(["ID_LIEU", "IS_AOUT"])["NB_VALD_TOTAL"].mean().unstack()
    profiles["creux_estival"] = profiles["ID_LIEU"].map(ete[True] / ete[False])

    profiles["log_vald"] = np.log1p(profiles["nb_vald_moyen"])
    profiles["cv"] = profiles["nb_vald_std"] / profiles["nb_vald_moyen"]

    return profiles.dropna().reset_index(drop=True)


def cluster_stations(df, n=N_CLUSTERS_DEFAULT, random_state=42):
    """Clusterise les stations par KMeans. Renvoie [ID_LIEU, cluster]."""
    profiles = build_station_profiles(df)

    X = profiles[["log_vald", "cv", "ratio_we_sem",
                  "ratio_vac_horsvac", "creux_estival"]]
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n, random_state=random_state, n_init=10)
    profiles["cluster"] = km.fit_predict(X_scaled)

    return profiles[["ID_LIEU", "cluster"]]


def add_clusters(df, n=N_CLUSTERS_DEFAULT, random_state=42):
    """Ajoute la colonne 'cluster' au dataset journalier (NaN si station ecartee)."""
    cluster_map = cluster_stations(df, n=n, random_state=random_state)
    return df.merge(cluster_map, on="ID_LIEU", how="left")


# --------------------------------------------------------------------------- #
# 4. Orchestrateur commun
# --------------------------------------------------------------------------- #
def build_features(df, with_cluster=False, n_clusters=N_CLUSTERS_DEFAULT):
    """Construit le DataFrame enrichi COMMUN aux deux modeles.

    Enchaine : features temporelles -> features meteo -> [cluster optionnel].
    NE fait PAS de scaling, NE cree PAS de lags (specifiques XGBoost).

    Parametres
    ----------
    df : DataFrame brut issu de data.get_data().
    with_cluster : si True, ajoute la colonne 'cluster'.
    n_clusters : nombre de clusters si with_cluster.

    Renvoie
    -------
    DataFrame au grain (JOUR, ID_LIEU), enrichi, non scale.
    """
    df = df.copy()
    df["JOUR"] = pd.to_datetime(df["JOUR"])

    df = add_time_features(df)
    df = add_weather_features(df)
    if with_cluster:
        df = add_clusters(df, n=n_clusters)

    return df
