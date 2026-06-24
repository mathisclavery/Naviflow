from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from naviflow.registry_xgb import load_global_bundle, load_global_features
from naviflow.config import TRAIN_FROM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Charge LE modèle global (un seul) + niveaux + features de service."""
    bundle = load_global_bundle(train_from=TRAIN_FROM)
    app.state.model = bundle["model"]
    app.state.levels = bundle["levels"]
    app.state.feature_names = bundle["feature_names"]
    app.state.horizon = bundle["meta"]["horizon"]
    app.state.features = load_global_features(train_from=TRAIN_FROM)


@app.get("/predict")
def predict(station_id: int, prediction_date: str):
    if station_id not in app.state.features or station_id not in app.state.levels.index:
        raise HTTPException(status_code=404, detail=f"No model data for station {station_id}")

    X_feat = app.state.features[station_id]
    mask = X_feat["JOUR"] == prediction_date
    if not mask.any():
        raise HTTPException(status_code=404,
                            detail=f"Date {prediction_date} not available for station {station_id}")

    # Le modèle prédit un RATIO (~1) par horizon ; on le repasse en absolu × niveau.
    X_pred = X_feat[mask][app.state.feature_names]
    ratio = app.state.model.predict(X_pred)[0]
    level = float(app.state.levels[station_id])

    return {
        "station_id": station_id,
        "prediction_date": prediction_date,
        "predictions": {f"J+{h + 1}": float(ratio[h] * level) for h in range(app.state.horizon)},
    }


@app.get('/ping')
def ping():
    return {'response': 'pong'}
