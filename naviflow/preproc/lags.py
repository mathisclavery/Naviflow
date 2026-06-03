"""Création des features de lag temporel par station."""

import pandas as pd


def add_lags(df, lags=(1, 7, 30), target='NB_VALD_TOTAL', group='ID_LIEU', date='JOUR'):
    """
    Ajoute des features de lag temporel, calculées par station.

    Les lags représentent l'affluence passée (J-1, J-7, J-30) et sont
    de loin les features les plus prédictives pour ce problème.

    Le groupby(station) est crucial : sans lui, le lag de la dernière
    ligne d'une station prendrait la première ligne de la station suivante.

    Args:
        df: DataFrame brut (issu de load())
        lags: tuple des décalages à créer en jours (défaut 1, 7, 30)
        target: colonne cible à décaler
        group: colonne de regroupement (station)
        date: colonne de date pour le tri

    Returns:
        DataFrame avec les colonnes lag_X ajoutées et les NaN supprimés
    """
    df = df.copy()
    df[date] = pd.to_datetime(df[date])
    df = df.sort_values([group, date])

    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(group)[target].shift(lag)

    df = df.dropna()

    return df