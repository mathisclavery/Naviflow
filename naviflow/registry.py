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

def save_results() -> None:
    pass


############################
# SAVE MODEL
############################

def save_model(model: keras.Model) -> None:
    pass


############################
# LOAD MODEL
############################

def load_model() -> keras.Model:
    pass
