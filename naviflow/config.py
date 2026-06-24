"""Configuration centralisee de Naviflow.

Tout ce qui peut varier (annees, encodages, noms de colonnes) vit ici, en un
seul endroit. C'est le premier fichier a editer pour reconfigurer le pipeline.
"""

import os
from pathlib import Path

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
MODEL_TARGET = os.getenv("MODEL_TARGET")

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
# Racine du projet = dossier parent du package naviflow/.
# Les chemins sont independants du dossier depuis lequel tu lances tes notebooks.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "raw_data"

# --------------------------------------------------------------------------- #
# Detection de format
# --------------------------------------------------------------------------- #
# Encodages et separateurs testes lors de la detection automatique.
ENCODINGS = ["utf-8-sig", "utf-16", "cp1252"]
SEPARATORS = ["\t", ";"]

# Pour chaque annee : liste de (periode, extension).
YEAR_CONFIG = {
    2015: [("S1", "csv"), ("S2", "csv")],
    2016: [("S1", "txt"), ("S2", "txt")],
    2017: [("S1", "txt"), ("S2", "txt")],
    2018: [("S1", "txt"), ("S2", "txt")],
    2019: [("S1", "txt"), ("S2", "txt")],
    2020: [("S1", "txt"), ("S2", "txt")],
    2021: [("S1", "txt"), ("S2", "txt")],
    2022: [("S1", "txt"), ("S2", "txt")],
    2023: [("S1", "txt"), ("S2", "txt")],
    2024: [("S1", "txt"), ("T3", "txt"), ("T4", "txt")],
    2025: [("T1", "csv"), ("T2", "csv"), ("T3", "csv"), ("T4", "csv")],
}

# --------------------------------------------------------------------------- #
# IDFM
# --------------------------------------------------------------------------- #
# Colonnes minimales qu'un fichier de validations valide doit contenir.
EXPECTED_COLS = {
    "JOUR", "CODE_STIF_TRNS", "CODE_STIF_RES", "CODE_STIF_ARRET",
    "LIBELLE_ARRET", "CATEGORIE_TITRE", "NB_VALD",
}

# La colonne identifiant de lieu change de nom selon les annees -> ID_LIEU.
ID_RENAME_MAP = {"ID_REFA_LDA": "ID_LIEU", "lda": "ID_LIEU", "ID_ZDC": "ID_LIEU"}

# Colonnes conservees apres lecture (avant agregation).
KEEP_COLS = ["JOUR", "ID_LIEU", "LIBELLE_ARRET", "CATEGORIE_TITRE", "NB_VALD"]

# Suffixe du nom de fichier : {year}-{period}-validations.{ext}
FILE_KIND = "validations"

# Valeurs sentinelles dans ID_LIEU
SENTINELLES = [-1, 0, 999999]

# Seuil qui determine si les deux ID se chevauchent ou se succèdent dans le temps
THRESHOLD_DAYS = 7

#Date à partir de laquelle on prend les données
TRAIN_FROM = "2015-01-01"

#Dates covid à exclure
EXCLUDE_WINDOW = ["2020-01-01", "2023-01-01"]

# Cutoff de DÉPLOIEMENT : le modèle de prod s'entraîne sur les données AVANT cette
# date, et la démo (features de service) ne sert que les dates À PARTIR de cette
# date. Ainsi la démo prédit des jours JAMAIS vus à l'entraînement (pas de leakage),
# et la perf affichée correspond au test held-out de l'évaluation. Mettre à None
# pour un déploiement « live » (réentraînement sur tout, prédiction du vrai futur).
DEPLOY_TEST_CUTOFF = "2025-05-26"

# Ratio qui determine si les deux ID qui se chevauchent sont complémentaires ou doublons.
RATIO_DOUBLON = 0.65

# Force la station juvisy à merge les ID en considerant que ce sont des doublons
STATION_OVERRIDES = {
    "juvisy": "merge_max",
}

ID_TO_DROP = [65227, 72641, 72787, 62667,  62939,  63244,  60856, 486998,  60776,  60387, 486996, 486999,
 486959,  60892,  62977, 486992,  60985,  60450,  60915, 64566,  73602, 480927, 480950, 480952,  64246,  64382,  73731,  64622,
  64049,  64589, 480951, 68293, 68311, 68505, 68582, 72028, 72033, 73312, 73334, 73360, 73411, 73709, 411281, 411284, 63284, 61327, 63278]

# Poles d'echange : regroupement de stations co-localisees
# Matching par cle de libelle EXACTE. Chaque pole somme ses stations.
POLES_DEFINITION = {
    "POLE_CHATELET": ["chatelet", "chatelet les halles", "les halles"],
    "POLE_LA_DEFENSE": ["defense", "la defense", "la defense grande arche"],
    "POLE_SAINT_LAZARE": ["auber", "haussmann saint lazare", "havr caumartin",
                          "havre caumartin", "saint lazare", "opera", "augustin"],
    "POLE_MONTPARNASSE": ["montparnasse"],
    "POLE_GARE_DE_LYON": ["gare de lyon"],
    "POLE_GARE_DU_NORD": ["gare du nord", "magenta"],
    "POLE_GARE_DE_LEST": ["gare de l est", "landon"],
}

# Base des IDs synthetiques attribues aux poles (negatifs pour ne pas
# collisionner avec les vrais ID_LIEU). 1er pole = -1000, 2e = -1001, etc.
POLE_ID_BASE = -1000

# --------------------------------------------------------------------------- #
# Meteo
# --------------------------------------------------------------------------- #

STATION_MONTSOURIS = 75114001
COLS_UTILES = ["AAAAMMJJ", "RR", "TN", "TX", "TM", "FFM"]
METEO_FILES = [
    "meteo/data-1950-2024-meteo/1950-2024_RR-T-Vent.csv",
    "meteo/data-2025-2026-meteo/2025-2026_RR-T-Vent.csv",
]

# --------------------------------------------------------------------------- #
# Calendrier
# --------------------------------------------------------------------------- #

START_YEAR = 2015
END_YEAR = 2026
ZONE = "C"  # Paris


# --------------------------------------------------------------------------- #
# RNN model constants
# --------------------------------------------------------------------------- #

######################################################
#CONSTANTS for RNN MODEL? Will depend on the feature engineering
#TODO: CR: Move relevant variables into config.py - Not sure yet

#########
#Define nature of each variable
PAST_COVARIATES = ['_RR', '_TN', '_TX', '_TM', '_FFM','_month_sin','_month_cos']
FUTURE_COVARIATES = ['_IS_WEEKEND', '_IS_FERIE', '_IS_VACANCES', '_IS_PONT']
TARGET = 'NB_VALD_TOTAL'
N_TARGETS = 1
N_FEATURES = 12 #Q-CR: Can we parametrize it directly as a function of the dataframe

####################################################
#DATASET SPLIT FOR MODEL

#####
# FOLDS
# NB: OPTIONAL FOR CROSS VALIDATION of the RNN model
# --------------------------------------------------- #
FOLD_LENGTH = 1*365*3 # 1 measure every day
                        # three years
# --------------------------------------------------- #
# Let's consider FOLDS starting every trimester       #
# --------------------------------------------------- #
FOLD_STRIDE = 1*91 # 1 measure every day
                   # 1 quarter = 91 days


######################################################
# --------------------------------------------------- #
# Let's consider a train-test-split ratio of 2/3      #
# --------------------------------------------------- #
TRAIN_TEST_RATIO = 0.66


#######################################################
NUMBER_STATIONS = 708 #CR: All stations 20260606 = 708 (after removing Tram stations)

#######################################################
#X, y shapes variables

#Number of past days before predict
INPUT_LENGTH = 140
#Number of days to predict/horizon
#NB: Assume no incubation/no gap between past and prediction days
OUTPUT_LENGTH = 7

########################################################
#Number of samples to take for training
N_TRAIN = 5000 # number_of_sequences_train = Samples
N_TEST =  int(N_TRAIN*(1-TRAIN_TEST_RATIO)) #To keep same proportion of samples to take in the TEST set


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #

N_CLUSTERS_DEFAULT = 4
