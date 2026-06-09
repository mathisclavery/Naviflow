from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from naviflow.registry_xgb import load_all_models, load_all_features

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
    app.state.models = load_all_models()
    app.state.features = load_all_features()

@app.get("/predict")
def predict(station_id: int, prediction_date: str):
    if station_id not in app.state.models:
        raise HTTPException(status_code=404, detail=f"No model for station {station_id}")

    model = app.state.models[station_id]
    X_test = app.state.features[station_id]

    mask = X_test["JOUR"] == prediction_date
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"Date {prediction_date} not available for station {station_id}")

    X_pred = X_test[mask].drop(columns=["JOUR"])
    y_pred = model.predict(X_pred)

    return {
        "station_id": station_id,
        "prediction_date": prediction_date,
        "predictions": {f"J+{h+1}": float(y_pred[0][h]) for h in range(7)}
    }

@app.get('/ping')
def ping():
    return {'response': 'pong'}
