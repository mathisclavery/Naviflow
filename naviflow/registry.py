"""Sauvegarde et chargement des modeles XGBoost (format natif) + metriques.

Strategie : un fichier modele par groupe (station ou cluster), au format natif
XGBoost (.json) plutot que pickle :
  - portable entre versions de la librairie,
  - plus leger (pas de surcouche pickle),
  - pas de risque d'execution de code au chargement.

Les metriques de tous les groupes sont consignees dans un seul CSV recapitulatif,
pour comparer les groupes sans recharger les modeles.

Arborescence produite :
    models_store/
        xgb_station_101.json
        xgb_station_202.json
        ...
        results_xgb.csv
"""

from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from naviflow.config import PROJECT_ROOT

# Dossier de stockage des modeles (cree au besoin)
MODELS_STORE = PROJECT_ROOT / "models_store"
RESULTS_CSV = MODELS_STORE / "results_xgb.csv"


def model_path(group_id, grain="station"):
    """Chemin du fichier modele pour un groupe donne.

    grain : 'station' ou 'cluster' — sert juste a nommer le fichier.
    """
    return MODELS_STORE / f"xgb_{grain}_{group_id}.json"


def save_model(model, group_id, grain="station"):
    """Sauvegarde un XGBRegressor au format natif XGBoost.

    Cree models_store/ si absent. Renvoie le chemin ecrit.
    """
    MODELS_STORE.mkdir(parents=True, exist_ok=True)
    path = model_path(group_id, grain)
    model.save_model(path)
    return path


def load_model(group_id, grain="station"):
    """Recharge un XGBRegressor depuis son fichier natif.

    Ne charge QUE le modele demande (utile pour une demo : on charge la seule
    station cliquee, pas les 743).
    """
    path = model_path(group_id, grain)
    if not path.exists():
        raise FileNotFoundError(f"Aucun modele sauvegarde pour {grain} {group_id} : {path}")
    model = XGBRegressor()
    model.load_model(path)
    return model


def save_results(results, grain="station"):
    """Consigne les metriques par groupe dans un CSV recapitulatif unique.

    Parametres
    ----------
    results : dict {group_id: {'mae': ..., 'r2': ..., ...}} ou
              liste de dicts contenant au moins 'group_id'.
    grain : 'station' ou 'cluster' (colonne ajoutee au CSV).

    Renvoie le DataFrame ecrit.
    """
    MODELS_STORE.mkdir(parents=True, exist_ok=True)

    if isinstance(results, dict):
        rows = [{"group_id": gid, **metrics} for gid, metrics in results.items()]
    else:
        rows = list(results)

    df = pd.DataFrame(rows)
    df.insert(0, "grain", grain)
    df.to_csv(RESULTS_CSV, index=False)
    return df
