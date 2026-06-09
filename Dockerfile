FROM python:3.10.6-buster

COPY naviflow naviflow
COPY requirements.txt requirements.txt
COPY setup.py setup.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install .

CMD uvicorn naviflow.api.api:app --host 0.0.0.0 --port $PORT
