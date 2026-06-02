"""Pipeline de validations : du fichier brut au volume journalier par station.

Fonction principale : `load()`. Elle enchaine detection de format, lecture,
nettoyage et agregation pour produire un DataFrame au grain (JOUR, ID_LIEU)
avec le nombre total de validations.
"""

import pandas as pd

from naviflow.sources import loaders
from naviflow.sources.stations import merge_stations
from naviflow.config import (DATA_DIR,
                             EXPECTED_COLS,
                             ID_RENAME_MAP,
                             KEEP_COLS,
                             SENTINELLES)


def select_columns(df, keep_cols=KEEP_COLS):
    """Ne conserve que les colonnes utiles.

    Les colonnes CODE_STIF_TRNS / RES / ARRET sont ecartees : codification
    transporteur non perenne dans le temps.
    """
    return df[[c for c in keep_cols if c in df.columns]].copy()


def clean_types(df):
    """Convertit les colonnes vers les bons types.

    NB_VALD, nettoyage en 3 temps :
      1. retrait de l'espace separateur de milliers ('2 093' -> '2093'),
         present notamment sur 2024-T3 ;
      2. 'Moins de 5' (volume censure 1-4) -> 3 ;
      3. conversion numerique (coerce pour les vrais vides residuels).
    """
    df = df.copy()
    df["JOUR"] = pd.to_datetime(df["JOUR"], format="mixed", dayfirst=True)
    df["ID_LIEU"] = pd.to_numeric(df["ID_LIEU"], errors="coerce").astype("Int64")

    nb = df["NB_VALD"]
    # Texte = object OU string natif (pandas 2.x peut typer en 'str'/'string').
    if nb.dtype == object or pd.api.types.is_string_dtype(nb):
        nb = (
            nb.str.replace(r"\s", "", regex=True)   # '2 093' -> '2093'
              .replace("Moinsde5", 3)               # l'espace a aussi ete retire ici
        )
    df["NB_VALD"] = pd.to_numeric(nb, errors="coerce")
    return df


def concat(data):
    """Aplatit le dict {annee: {periode: df}} en un DataFrame unique, nettoye et
    tracable (colonnes ANNEE, PERIODE ajoutees)."""
    frames = []
    for year, periods in data.items():
        for period, df in periods.items():
            df = df.rename(columns=ID_RENAME_MAP)
            df = select_columns(df)
            df = clean_types(df)
            df["ANNEE"] = year
            df["PERIODE"] = period
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate_by_station(df):
    """Agrege au grain (jour x station) en sommant toutes les categories de titre.
    NB_VALD devient l'affluence totale de la station ce jour-la.

    Les lignes sans ID_LIEU ou sans NB_VALD sont ecartees : inexploitables pour
    une serie temporelle par station.
    """
    df = df.dropna(subset=["ID_LIEU", "NB_VALD"])
    return (
        df.groupby(["JOUR", "ID_LIEU"], as_index=False)
          .agg(
              NB_VALD_TOTAL=("NB_VALD", "sum"),
              LIBELLE_ARRET=("LIBELLE_ARRET", "first"),
          )
    )


def drop_sentinelles(df, sentinelles=SENTINELLES):
    """Retire les lignes a ID_LIEU sentinelle.

    - ID_LIEU -1 / 0 : lieu non identifie.
    - ID_LIEU 999999 : agregat reseau (pas une station individuelle).
    """
    return df[~df["ID_LIEU"].isin(sentinelles)].reset_index(drop=True)


def normalize_labels(df):
    """[Etape 3] Normalise LIBELLE_ARRET.
    """
    df = df.copy()
    s = df["LIBELLE_ARRET"].astype(str)
    s = s.str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.title()
    df["LIBELLE_ARRET"] = s
    return df


def clean(df):
    """Enchaine les etapes de nettoyage post-agregation.
    """
    df = drop_sentinelles(df)
    df = normalize_labels(df)
    df = merge_stations(df)
    return df


def load_validations(data_dir=DATA_DIR, aggregate=True, verbose=False):
    """Pipeline complet, de bout en bout.

    Parametres
    ----------
    data_dir : dossier racine des donnees (defaut : config.DATA_DIR).
    aggregate : si True (defaut), renvoie le grain (JOUR, ID_LIEU, NB_VALD_TOTAL).
        Si False, renvoie le DataFrame nettoye au grain ligne par categorie de titre.
    verbose : affiche la detection des formats.

    Renvoie un DataFrame pandas.
    """
    formats = loaders.detect_formats(data_dir, EXPECTED_COLS, verbose=verbose)
    data = loaders.load_source(data_dir, formats, EXPECTED_COLS)
    df = concat(data)
    if not aggregate:
        return df
    df = aggregate_by_station(df)
    df = clean(df)

    return df


def run_quality_checks(df_raw, df_agg):
    """Controles d'integrite sur le DataFrame nettoye et l'agrege."""
    print("=== Apres nettoyage (df concatene) ===")
    print(f"Lignes             : {len(df_raw):,}")
    print(f"JOUR non convertis : {df_raw['JOUR'].isna().sum():,}")
    print(f"ID_LIEU manquants  : {df_raw['ID_LIEU'].isna().sum():,}")
    print(f"NB_VALD = NaN      : {df_raw['NB_VALD'].isna().sum():,}")
    print(f"NB_VALD min / max  : {df_raw['NB_VALD'].min()} / {df_raw['NB_VALD'].max()}")
    print(f"Periode            : {df_raw['JOUR'].min():%Y-%m-%d} -> {df_raw['JOUR'].max():%Y-%m-%d}")

    print("\n=== Apres agregation + nettoyage (jour x station) ===")
    print(f"Lignes             : {len(df_agg):,}")
    print(f"Stations distinctes: {df_agg['ID_LIEU'].nunique():,}")
    print(f"NB_VALD_TOTAL min  : {df_agg['NB_VALD_TOTAL'].min()}  (doit etre > 0)")
    sentinelles_restantes = df_agg["ID_LIEU"].isin(SENTINELLES).sum()
    print(f"Sentinelles restantes: {sentinelles_restantes:,}  (doit valoir 0)")
    doublons = df_agg.duplicated(subset=["JOUR", "ID_LIEU"]).sum()
    print(f"Doublons (jour,stat): {doublons:,}  (doit valoir 0)")
    assert doublons == 0, "Doublons detectes apres agregation"

    print("\n=== Stations distinctes par annee ===")
    print(df_agg.groupby(df_agg["JOUR"].dt.year)["ID_LIEU"].nunique())
