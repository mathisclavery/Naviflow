import os

import pandas as pd

from naviflow.ml_logic.sources.validations import load_validations
from naviflow.ml_logic.sources.meteo import load_meteo
from naviflow.ml_logic.sources.calendrier import load_calendrier
from naviflow.config import TRAIN_FROM as DEFAULT_TRAIN_FROM

from naviflow.ml_logic.preprocess_rnn import create_rnn_dataframe

import json



def get_data(train_from=None):
    """Charge et joint toutes les données du projet.

    Combine les validations journalières par station avec les données météo
    de Paris-Montsouris et les données de calendrier (jointure left sur JOUR).

    Renvoie
    -------
    pd.DataFrame
        Colonnes : JOUR, ID_LIEU, NB_VALD_TOTAL, RR, TN, TX, TM, FFM, .
        Une ligne par (jour, station). La météo est dupliquée sur toutes les
        stations d'un même jour (météo unique par jour pour toute l'Île-de-France).
    """
    if train_from is None:
        train_from = os.getenv("TRAIN_FROM", DEFAULT_TRAIN_FROM)

    df_valid = load_validations()
    df_meteo = load_meteo()
    df_calendrier = load_calendrier()

    df = (df_valid
          .merge(df_meteo, on="JOUR", how="left")
          .merge(df_calendrier, on="JOUR", how="left"))

    df["JOUR"] = pd.to_datetime(df["JOUR"])
    df = df[df["JOUR"] >= pd.to_datetime(train_from)].reset_index(drop=True)

    return df


def create_stations_dict(df,y):
    """
    df: original dataframe
    y: target keras tensorflow - It last dimension shape is the number of stations to predict
    """

    #Simple way found to extract list of station columns
    df_stations = create_rnn_dataframe(df,log=True)
    list_id_lieu = df_stations.columns[:y.shape[2]]

    dict_stations = {list_id_lieu[i] : df[df['ID_LIEU'] == list_id_lieu[i]].groupby(['LIBELLE_ARRET']).count().iloc[0].name for i in range(len(list_id_lieu))}

    return dict_stations



def save_stations_dict(dict):
    with open('dict_stations.json', 'w') as f:
        json.dump(dict, f)

