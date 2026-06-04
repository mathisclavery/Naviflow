"""Chargement des données météo Paris-Montsouris."""

from pathlib import Path

import pandas as pd

from naviflow.config import DATA_DIR, COLS_UTILES, METEO_FILES, STATION_MONTSOURIS

def _read_one(path):
    """Lit un fichier météo brut sans transformation."""
    return pd.read_csv(DATA_DIR / path, sep=";", low_memory=False)


def load_meteo():
    """Charge les données météo journalières de la station Paris-Montsouris.

    Concatène les deux fichiers Météo-France (historique 1950-2024 et récent
    2025-2026), filtre sur la station Paris-Montsouris (NUM_POSTE 75114001),
    et ne garde que les variables utiles à la prédiction d'affluence sur la
    période 2015 -> aujourd'hui.

    La période n'est PAS bornée à droite : c'est la jointure ultérieure avec
    les validations qui s'occupera de couper si besoin.

    Renvoie
    -------
    pd.DataFrame
        Colonnes : JOUR (datetime), RR (pluie mm), TN/TX/TM (températures °C),
        FFM (vent moyen m/s). Une ligne par jour.
    """
    # 1. Lire et concaténer les deux fichiers d'un coup
    df = pd.concat([_read_one(p) for p in METEO_FILES], ignore_index=True)

    # 2. Filtrer Montsouris
    df = df[df["NUM_POSTE"] == STATION_MONTSOURIS]

    # 3. Convertir la date et filtrer la période utile pour le projet
    df["AAAAMMJJ"] = pd.to_datetime(df["AAAAMMJJ"], format="%Y%m%d")
    df = df[df["AAAAMMJJ"].dt.year >= 2015]

    # 4. Garder les colonnes utiles et renommer la date
    df = df[COLS_UTILES].rename(columns={"AAAAMMJJ": "JOUR"})

    return df.reset_index(drop=True)
