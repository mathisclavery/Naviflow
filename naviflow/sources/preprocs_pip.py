# ──────────────────────────────────────────────────────────────
# CONSTRUCTION DU PIPELINE
# ──────────────────────────────────────────────────────────────

# Colonnes continues à scaler (brutes + amplitude)
weather_continuous = ["RR", "TN", "TX", "TM", "FFM", "amplitude_thermique"]

# Colonnes binaires dérivées (pas de scaling)
weather_binary = ["is_rain", "is_wind", "is_cold", "is_hot", "meteo_degradee"]

# La branche météo est elle-même un ColumnTransformer imbriqué :
#   ① transform_weather_features crée toutes les colonnes
#   ② RobustScaler sur les continues, passthrough sur les binaires
weather_transformer = Pipeline(steps=[
    ("engineering", FunctionTransformer(transform_weather_features, validate=False)),
    ("scaling", ColumnTransformer(
        transformers=[
            ("scale",   RobustScaler(), weather_continuous),
            ("binary",  "passthrough",  weather_binary),
        ],
        remainder="drop"
    ))
])

# ColumnTransformer principal : 3 branches en parallèle
preprocessor = ColumnTransformer(
    transformers=[
        ("time",    FunctionTransformer(transform_time_features, validate=False), temporal_cols),
        ("weather", weather_transformer,                                          weather_cols),
        ("binary",  "passthrough",                                                binary_cols),
    ],
    remainder="drop"
)

# Pipeline complet (le modèle se branche à la suite)
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    # ("model", MonModele()),
])


# ──────────────────────────────────────────────────────────────
# 5. VÉRIFICATION RAPIDE
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = pd.DataFrame({
        "JOUR":         ["2015-01-01", "2015-07-14", "2015-12-31"],
        "ID_LIEU":      [59403, 59420, 59447],
        "LIBELLE_ARRET":["ANGERVILLE", "BOIGNEVILLE", "BUNO-GIRONVILLE"],
        "RR":  [0.0,  5.2,  0.3],
        "TN":  [-0.6, 14.1, 1.0],
        "TX":  [6.5,  32.0, 7.2],
        "TM":  [2.5,  23.0, 4.0],
        "FFM": [2.2,  45.0, 12.0],
        "IS_WEEKEND":  [False, False, True],
        "IS_FERIE":    [True,  True,  False],
        "IS_VACANCES": [True,  True,  False],
        "IS_PONT":     [False, False, False],
        "NB_VALD_TOTAL": [13.0, 6.0, 12.0],  # target — exclue de X
    })

    X = sample.drop(columns=["NB_VALD_TOTAL", "ID_LIEU", "LIBELLE_ARRET"])
    y = sample["NB_VALD_TOTAL"]

    result = preprocessor.fit_transform(X)

    # Noms des colonnes produites (pour lisibilité)
    col_names = (
        ["jour_semaine", "jour_mois", "jour_annee", "semaine_annee",
         "mois", "trimestre", "annee", "is_debut_mois", "is_fin_mois"]
        + weather_continuous
        + weather_binary
        + binary_cols
    )

    print(pd.DataFrame(result, columns=col_names).T)
