"""
Clustering des stations Naviflow par KMeans.

Construit un profil par station (niveau, variabilite, profils temporels)
a partir du dataset journalier, puis attribue chaque station a un cluster.

Usage typique :
    from naviflow.pipeline import load
    from naviflow.preproc.clustering import add_clusters

    df = load()
    df = add_clusters(df)            # n par defaut depuis config
    df = add_clusters(df, n=5)       # n choisi explicitement
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from naviflow.config import SENTINELLES, N_CLUSTERS_DEFAULT


def build_station_profiles(df):
    """Agrege le dataset journalier en un profil par station.

    Une ligne = une station, decrite par des features de niveau et de forme :
    log_vald, cv, ratio_we_sem, ratio_vac_horsvac, creux_estival.
    """

    # Features de base agregees
    profiles = df.groupby("ID_LIEU").agg(
        nb_vald_moyen=("NB_VALD_TOTAL", "mean"),
        nb_vald_std=("NB_VALD_TOTAL", "std"),
    ).reset_index()

    # Ratio weekend / semaine (IS_WEEKEND deja present dans le df)
    wk = df.groupby(["ID_LIEU", "IS_WEEKEND"])["NB_VALD_TOTAL"].mean().unstack()
    profiles["ratio_we_sem"] = profiles["ID_LIEU"].map(wk[True] / wk[False])

    # Ratio vacances / hors-vacances
    vac = df.groupby(["ID_LIEU", "IS_VACANCES"])["NB_VALD_TOTAL"].mean().unstack()
    profiles["ratio_vac_horsvac"] = profiles["ID_LIEU"].map(vac[True] / vac[False])

    # Creux estival (aout vs reste de l'annee) pour détecter les stations de quartier délaissées par les parisiens en aout
    df["IS_AOUT"] = df["JOUR"].dt.month.isin([8])
    ete = df.groupby(["ID_LIEU", "IS_AOUT"])["NB_VALD_TOTAL"].mean().unstack()
    profiles["creux_estival"] = profiles["ID_LIEU"].map(ete[True] / ete[False])

    # Transformations anti-skew + variabilite relative
    profiles["log_vald"] = np.log1p(profiles["nb_vald_moyen"])
    profiles["cv"] = profiles["nb_vald_std"] / profiles["nb_vald_moyen"]

    return profiles.dropna().reset_index(drop=True)


def cluster_stations(df, n=N_CLUSTERS_DEFAULT, random_state=42):
    """Clusterise les stations par KMeans.

    Args:
        df: dataset journalier (grain JOUR x ID_LIEU) issu de pipeline.load().
        n: nombre de clusters. Defaut : config.N_CLUSTERS_DEFAULT.
        random_state: graine pour la reproductibilite.

    Returns:
        DataFrame [ID_LIEU, cluster] : le mapping station -> cluster.
    """
    profiles = build_station_profiles(df)

    X = profiles[['log_vald', 'cv', 'ratio_we_sem', 'ratio_vac_horsvac', 'creux_estival']]
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n, random_state=random_state, n_init=10)
    profiles["cluster"] = km.fit_predict(X_scaled)

    return profiles[["ID_LIEU", "cluster"]]


def add_clusters(df, n=N_CLUSTERS_DEFAULT, random_state=42):
    """Ajoute la colonne 'cluster' au dataset journalier.

    Args:
        df: dataset journalier issu de pipeline.load().
        n: nombre de clusters. Defaut : config.N_CLUSTERS_DEFAULT.
        random_state: graine pour la reproductibilite.

    Returns:
        df enrichi d'une colonne 'cluster' (NaN pour les sentinelles ecartees).
    """
    cluster_map = cluster_stations(df, n=n, random_state=random_state)
    return df.merge(cluster_map, on="ID_LIEU", how="left")
