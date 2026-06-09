from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.preprocess_xgb import add_lags
from naviflow import registry_xgb
from naviflow.config import TRAIN_FROM

# --------------------------------------------------------------------------- #
# Initialisation FastAPI
# --------------------------------------------------------------------------- #

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Chargement des données au démarrage
# --------------------------------------------------------------------------- #

NON_FEATURE_COLS = ["JOUR", "ID_LIEU", "LIBELLE_ARRET", "NB_VALD_TOTAL"]  

print("Chargement des données...")
df = get_data()
df = build_features(df)
df = add_lags(df)
print(f"Données prêtes : {len(df):,} lignes")

app.state.df = df
app.state.models = {}  # cache des modèles par station_id

# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/")
def root():
    return {"status": "API Naviflow OK"}

@app.get("/ping")
def ping():
    return {"response": "pong"}

@app.get("/predict")
def predict(
    station_id: int,
    date: str,
    horizon: int = 7,
    grain: str = "station",
    train_from: str = TRAIN_FROM
):
    """
    Prédit la fréquentation J+1..J+horizon pour une station.

    Exemple :
    http://127.0.0.1:8000/predict?station_id=59403&date=2025-01-01
    """

    df = app.state.df

    # 1. Filtrer la station (ID_LIEU, pas station_id)
    station_df = df[df["ID_LIEU"] == station_id].copy()
    if station_df.empty:
        raise HTTPException(status_code=404, detail=f"Station {station_id} inconnue")

    # 2. Charger le modèle (avec cache)
    if station_id not in app.state.models:
        try:
            app.state.models[station_id] = registry_xgb.load_model(
                station_id,
                grain=grain,
                horizon=horizon,
                train_from=train_from
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Aucun modèle trouvé pour la station {station_id}"
            )

    model = app.state.models[station_id]

    # 3. Construire X : dernière ligne avant la date demandée
    station_df["JOUR"] = pd.to_datetime(station_df["JOUR"])
    station_df = station_df[station_df["JOUR"] <= date].sort_values("JOUR")

    if station_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune donnée disponible avant le {date} pour la station {station_id}"
        )

    feature_cols = [c for c in station_df.columns if c not in NON_FEATURE_COLS]
    X = station_df[feature_cols].tail(1)

    # 4. Prédire
    try:
        predictions = model.predict(X)[0]  # shape (horizon,)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    return {
        "station_id": station_id,
        "date": date,
        "predictions": {
            f"J+{h+1}": round(float(predictions[h]), 1)
            for h in range(horizon)
        }
    }
