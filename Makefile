# -------------------------
# NAVIFLOW — MLOps Makefile
# -------------------------

preprocess:
    python -m naviflow.interface.main preprocess

train:
    python -m naviflow.interface.main train

evaluate:
    python -m naviflow.interface.main evaluate

predict:
    python -m naviflow.interface.main pred

upload_raw:
    python -m naviflow.gcp.upload_raw_data

api:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

