FROM python:3.10.6-buster

# Dépendances SLIM de l'API (pas tensorflow/sklearn/plotting) -> image légère.
COPY requirements-api.txt requirements-api.txt
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# Code du package. requirements.txt est copié car setup.py le lit, mais on
# installe SANS dépendances (--no-deps) : elles sont déjà fournies, slim, ci-dessus.
COPY naviflow naviflow
COPY setup.py requirements.txt ./
RUN pip install --no-deps .

# Cloud Run fournit $PORT au runtime.
CMD uvicorn naviflow.api.api:app --host 0.0.0.0 --port $PORT
