import os
import time
import pickle
import glob
from tensorflow import keras
from google.cloud import storage

from naviflow.params import (
    LOCAL_REGISTRY_PATH,
    MODEL_TARGET,
    BUCKET_NAME
)

############################
# SAVE RESULTS
############################

def save_results(params: dict = None, metrics: dict = None) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", f"{timestamp}.pickle")
        with open(params_path, "wb") as f:
            pickle.dump(params, f)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", f"{timestamp}.pickle")
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics, f)

    print("✅ Results saved locally")

    # Save to GCS if needed
    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        if params is not None:
            blob = bucket.blob(f"params/{timestamp}.pickle")
            blob.upload_from_filename(params_path)

        if metrics is not None:
            blob = bucket.blob(f"metrics/{timestamp}.pickle")
            blob.upload_from_filename(metrics_path)

        print("☁️ Results also saved to GCS")


############################
# SAVE MODEL
############################

def save_model(model: keras.Model) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")

    model.save(model_path)
    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{timestamp}.h5")
        blob.upload_from_filename(model_path)
        print("☁️ Model also saved to GCS")


############################
# LOAD MODEL
############################

def load_model() -> keras.Model:
    model_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
    model_paths = glob.glob(f"{model_dir}/*.h5")

    if not model_paths:
        print("⚠️ No model found locally")
        return None

    latest = sorted(model_paths)[-1]
    print(f"📦 Loading model: {latest}")

    return keras.models.load_model(latest)
