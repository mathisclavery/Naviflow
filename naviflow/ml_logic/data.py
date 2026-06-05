from naviflow.ml_logic.sources.validations import load_validations
from naviflow.ml_logic.sources.meteo import load_meteo
from naviflow.ml_logic.sources.calendrier import load_calendrier


def get_data():
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
    df_valid = load_validations()
    df_meteo = load_meteo()
    df_calendrier = load_calendrier()

    df = (df_valid
          .merge(df_meteo, on="JOUR", how="left")
          .merge(df_calendrier, on="JOUR", how="left"))

    return df
