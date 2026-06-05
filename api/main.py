from fastapi import FastAPI
from naviflow.registry import load_model

app = FastAPI()
model = load_model("baseline_mean")

@app.get("/")
def root():
    return {"status": "Naviflow API running"}

@app.post("/predict")
def predict(payload: dict):
    X = payload["features"]
    y_pred = model.predict([X])[0]
    return {"prediction": y_pred}
