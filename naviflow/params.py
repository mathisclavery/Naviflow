import os

############################
# PROJECT CONFIG
############################

PROJECT = "naviflow"

############################
# LOCAL REGISTRY
############################

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_REGISTRY_PATH = os.path.join(BASE_DIR, "..", "naviflow_registry")

os.makedirs(os.path.join(LOCAL_REGISTRY_PATH, "models"), exist_ok=True)
os.makedirs(os.path.join(LOCAL_REGISTRY_PATH, "metrics"), exist_ok=True)
os.makedirs(os.path.join(LOCAL_REGISTRY_PATH, "params"), exist_ok=True)

############################
# MODEL TARGET
############################
# "local" → tout reste sur ton disque
# "gcs"   → artefacts envoyés dans ton bucket GCS
# "mlflow" → tracking MLflow
MODEL_TARGET = "local"

############################
# GCS CONFIG
############################

BUCKET_NAME = "naviflow-registry"  # tu changeras après création
