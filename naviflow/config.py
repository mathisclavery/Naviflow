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
DATA_DIR = PROJECT_ROOT / "data"

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
# Colonnes
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
