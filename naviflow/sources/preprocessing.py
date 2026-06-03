# ============================================================
# PREPROCESSING PIPELINE — Affluence du métro parisien
# ============================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ──────────────────────────────────────────────────────────────
# 1. DÉFINITION DES COLONNES PAR TYPE
# ──────────────────────────────────────────────────────────────
temporal_cols = ["JOUR"]
weather_cols  = ["RR", "TN", "TX", "TM", "FFM"]
binary_cols   = ["IS_WEEKEND", "IS_FERIE", "IS_VACANCES", "IS_PONT"]


# ──────────────────────────────────────────────────────────────
# 2. FEATURES TEMPORELLES
# ──────────────────────────────────────────────────────────────
def transform_time_features(X: pd.DataFrame) -> np.ndarray:
    """
    Extrait des features calendaires à partir de la colonne JOUR.
    Features produites (toutes ordinales, pas de cyclique) :
    ┌─────────────────┬──────────────────────────────────────────┐
    │ Feature          │ Valeurs                                 │
    ├─────────────────┼──────────────────────────────────────────┤
    │ jour_semaine     │ 0=lundi … 6=dimanche                    │
    │ jour_mois        │ 1 – 31                                  │
    │ jour_annee       │ 1 – 365/366                             │
    │ semaine_annee    │ 1 – 53  (numéro ISO de semaine)         │
    │ mois             │ 1 – 12                                  │
    │ trimestre        │ 1 – 4                                   │
    │ annee            │ ex. 2015                                │
    │ is_debut_mois    │ 1 si jour ≤ 5, sinon 0                  │
    │ is_fin_mois      │ 1 si jour ≥ 25, sinon 0                 │
    └─────────────────┴──────────────────────────────────────────┘

    Remarques :
    - jour_annee et semaine_annee capturent la saisonnalité "fine"
      sans nécessiter sin/cos (les modèles arborescents n'en ont
      pas besoin ; pour des réseaux de neurones on pourrait ajouter
      l'encodage cyclique en complément).
    - is_debut_mois / is_fin_mois signalent les pics de fréquentation
      liés aux flux domicile-travail en début et fin de mois.

    Paramètres
    ----------
    X : pd.DataFrame contenant au minimum la colonne "JOUR"
    Retourne ----- np.ndarray de shape (n_samples, 11)
    """
    dates = pd.to_datetime(X["JOUR"])

    jour_semaine  = dates.dt.dayofweek            # 0 = lundi
    jour_mois     = dates.dt.day                  # 1 – 31
    jour_annee    = dates.dt.dayofyear            # 1 – 366
    semaine_annee = dates.dt.isocalendar().week.astype(int)  # ISO week
    mois          = dates.dt.month                # 1 – 12
    trimestre     = dates.dt.quarter              # 1 – 4
    annee         = dates.dt.year

    is_debut_mois = (jour_mois <= 5).astype(int)
    is_fin_mois   = (jour_mois >= 25).astype(int)

    mois_sin = np.sin(2 * np.pi * mois / 12)
    mois_cos = np.cos(2 * np.pi * mois / 12)

    return np.column_stack([
        jour_semaine,
        jour_mois,
        jour_annee,
        semaine_annee,
        mois,
        trimestre,
        annee,
        is_debut_mois,
        is_fin_mois,
        mois_sin,
        mois_cos
    ])


# ──────────────────────────────────────────────────────────────
# 3. FEATURES MÉTÉO + NOUVELLES FEATURES DÉRIVÉES
# ──────────────────────────────────────────────────────────────
# Seuils (ajustables)
SEUIL_PLUIE  = 1.0   # mm/j  — pluie significative
SEUIL_VENT   = 40.0  # km/h  — vent fort
SEUIL_FROID  = 5.0   # °C    — froid ressenti
SEUIL_CHAUD  = 28.0  # °C    — chaleur significative

def transform_weather_features(X: pd.DataFrame) -> np.ndarray:
    """
    Prépare les features météo brutes ET crée des features dérivées.
    Features brutes conservées (après imputation médiane) :
      RR, TN, TX, TM, FFM
    Features dérivées créées :
    ┌───────────────────┬───────────────────────────────────────────────┐
    │ Feature            │ Définition                                   │
    ├───────────────────┼───────────────────────────────────────────────┤
    │ amplitude_thermique│ TX – TN  (confort ressenti dans la journée)  │
    │ is_rain            │ 1 si RR ≥ SEUIL_PLUIE                        │
    │ is_wind            │ 1 si FFM ≥ SEUIL_VENT                        │
    │ is_cold            │ 1 si TM ≤ SEUIL_FROID                        │
    │ is_hot             │ 1 si TM ≥ SEUIL_CHAUD                        │
    │ meteo_degradee     │ 1 si is_rain OU is_wind OU is_cold OU is_hot │
    └───────────────────┴───────────────────────────────────────────────┘

    Note sur le scaling :
    - RobustScaler est appliqué UNIQUEMENT sur les features continues
      (RR, TN, TX, TM, FFM, amplitude_thermique).
    - Les features binaires dérivées (is_*, meteo_degradee) sont
      laissées en 0/1 — les scaler n'apporterait rien.

    Paramètres
    ----------
    X : pd.DataFrame avec les colonnes [RR, TN, TX, TM, FFM]

    Retourne
    --------
    pd.DataFrame de shape (n_samples, 11)
    Les colonnes continues sont en tête pour faciliter le slicing
    dans le pipeline.
    """
 # Copie défensive + conversion float
    X = X.copy().astype(float)

    # Imputation médiane vectorisée
    X = X.fillna(X.median())

    # Features continues dérivées
    amplitude = X["TX"] - X["TN"]
    log_RR    = np.log1p(X["RR"])

    # Features binaires dérivées
    is_rain = (X["RR"]  >= SEUIL_PLUIE).astype(int)
    is_wind = (X["FFM"] >= SEUIL_VENT).astype(int)
    is_cold = (X["TM"]  <= SEUIL_FROID).astype(int)
    is_hot  = (X["TM"]  >= SEUIL_CHAUD).astype(int)

    # Score météo ordinal (0–4)
    meteo_score = is_rain + is_wind + is_cold + is_hot

    # Météo dégradée binaire (au moins un problème)
    meteo_degradee = (meteo_score > 0).astype(int)

    # Construction finale — ordre logique :
    # 1) features continues brutes
    # 2) features continues dérivées
    # 3) features binaires
    return np.column_stack([
        X["RR"].values,
        X["TN"].values,
        X["TX"].values,
        X["TM"].values,
        X["FFM"].values,
        amplitude.values,
        log_RR.values,
        is_rain.values,
        is_wind.values,
        is_cold.values,
        is_hot.values,
        meteo_score.values,
        meteo_degradee.values,
    ])


# ============================================================
# PREPROCESSING PIPELINE — Affluence du métro parisien
# ============================================================
# --- Transformers sklearn pour nos fonctions custom ---
time_transformer = FunctionTransformer(
    transform_time_features,
    validate=False
)

weather_transformer = FunctionTransformer(
    transform_weather_features,
    validate=False
)

# --- Scaler pour les features météo continues ---
# Les 7 premières colonnes météo sont continues :
# RR, TN, TX, TM, FFM, amplitude, log_RR
weather_continuous_scaler = RobustScaler()

# --- ColumnTransformer principal ---
preprocessor = ColumnTransformer(
    transformers=[
        # 1) Features temporelles → 11 colonnes
        ("time", time_transformer, temporal_cols),

        # 2) Features météo → 13 colonnes (dont 7 continues)
        ("weather_raw", weather_transformer, weather_cols),

        # 3) Scaling des 7 premières colonnes météo
        ("weather_scale", weather_continuous_scaler, slice(11, 18)),

        # 4) Variables binaires (déjà 0/1)
        ("binary", "passthrough", binary_cols),
    ],
    remainder="drop"
)

def preprocess_to_dataframe(preprocessor, X):
    """
    Applique le preprocessing sklearn et renvoie un DataFrame
    avec les noms de colonnes reconstruits.
    """
    # 1. Transformation
    X_preprocessed = preprocessor.fit_transform(X)
    # 2. Reconstruction des noms de colonnes
    time_feature_names = [
        "jour_semaine", "jour_mois", "jour_annee", "semaine_annee",
        "mois", "trimestre", "annee",
        "is_debut_mois", "is_fin_mois",
        "mois_sin", "mois_cos"
    ]
    weather_feature_names = [
        "RR", "TN", "TX", "TM", "FFM",
        "amplitude", "log_RR",
        "is_rain", "is_wind", "is_cold", "is_hot",
        "meteo_score", "meteo_degradee"
    ]
    binary_cols = ["IS_WEEKEND", "IS_FERIE", "IS_VACANCES", "IS_PONT"]
    final_columns = (
        time_feature_names +
        weather_feature_names +
        binary_cols
    )
    # 3. Conversion en DataFrame
    return pd.DataFrame(X_preprocessed, columns=final_columns)
