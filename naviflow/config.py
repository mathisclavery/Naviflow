"""Configuration centralisee de Naviflow.

Tout ce qui peut varier (annees, encodages, noms de colonnes) vit ici, en un
seul endroit. C'est le premier fichier a editer pour reconfigurer le pipeline.
"""

from pathlib import Path

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

# Ratio qui determine si les deux ID qui se chevauchent sont complémentaires ou doublons.
RATIO_DOUBLON = 0.65

# Force la station juvisy à merge les ID en considerant que ce sont des doublons
STATION_OVERRIDES = {
    "juvisy": "merge_max",
}

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

########################################
#CONSTANTS for RNN MODEL? Will depend on the feature engineering
#TODO: CR: Move relevant variables into config.py - Not sure yet

#########
#Define nature of each variable
PAST_COVARIATES = ['RR', 'TN', 'TX', 'TM', 'FFM','mois_sin','mois_cos']
FUTURE_COVARIATES = ['IS_WEEKEND', 'IS_FERIE', 'IS_VACANCES', 'IS_PONT']
TARGET = 'NB_VALD_TOTAL'
N_TARGETS = 1
N_FEATURES = 12 #Q-CR: Can we parametrize it directly as a function of the dataframe

#########
#DATASET SPLIT FOR MODEL

# FOLDS
# --------------------------------------------------- #
FOLD_LENGTH = 1*365*3 # 1 measure every day
                        # three years
# --------------------------------------------------- #
# Let's consider FOLDS starting every trimester       #
# --------------------------------------------------- #
FOLD_STRIDE = 1*91 # 1 measure every day
                   # 1 quarter = 91 days
# --------------------------------------------------- #
# Let's consider a train-test-split ratio of 2/3      #
# --------------------------------------------------- #
TRAIN_TEST_RATIO = 0.66

#Number of past days before predict
INPUT_LENGTH = 7

#Number of days to predict
#NB: Assume no incubation/no gap between past and prediction days
OUTPUT_LENGTH = 1


N_TRAIN = 1000 # number_of_sequences_train -
N_TEST =  333


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #

N_CLUSTERS_DEFAULT = 4
