"""Pipeline d'entrainement XGBoost de bout en bout.

Enchaine : get_data -> build_features -> boucle sur les groupes -> prepare_xgb
(en numpy) -> entrainement -> sauvegarde modele + metriques.

Le GRAIN d'entrainement est parametrable :
  - grain='station' : un modele par station (743 modeles).
  - grain='cluster' : un modele par cluster (4 modeles), chaque modele voyant
                      toutes les stations de son cluster.

Memoire : le gros DataFrame de toutes les stations n'est JAMAIS converti en un
seul numpy array. On le decoupe par groupe (chaque groupe fait quelques milliers
de lignes), on convertit ce petit morceau en numpy juste avant fit, et on libere
apres chaque iteration.
"""

import gc

from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.preprocess_xgb import prepare_xgb
from naviflow.ml_logic.models.sklearn_models import run_xgboost
from naviflow import registry


def train_all(grain="station", n_clusters=4, lags=(1, 7, 30), horizon=None,
              n_iter=50, save=True, verbose=True):
    """Entraine un XGBoost par groupe (station ou cluster) et sauvegarde tout.

    Parametres
    ----------
    grain : 'station' (boucle sur ID_LIEU) ou 'cluster' (boucle sur cluster).
    n_clusters : nombre de clusters si grain='cluster' (alimente build_features).
    lags : decalages temporels passes a prepare_xgb.
    horizon : si renseigne, entraine a predire J+horizon (sinon jour courant).
    n_iter : iterations du RandomizedSearch dans run_xgboost.
    save : si True, sauvegarde chaque modele + le CSV de metriques.
    verbose : log la progression.

    Renvoie
    -------
    dict {group_id: metrics} (metriques par groupe, sans les modeles).
    """
    Upload les données brutes vers GCS.
    """
    import naviflow.gcp.upload_raw_data as upload_raw_data
    upload_raw_data.upload_folder(
        upload_raw_data.LOCAL_FOLDER,
        upload_raw_data.BUCKET_FOLDER
    )

run_upload_raw:
    python -m naviflow.interface.main upload_raw


def preprocess():
    ...

def train():
    ...

def evaluate():
    ...

def pred():
    ...

def upload_raw():
    ...
