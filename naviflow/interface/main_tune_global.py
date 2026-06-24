"""Tuning des hyperparamètres du modèle GLOBAL poolé.

Lance une recherche aléatoire (GLOBAL_PARAM_GRID) sur le modèle poolé et fige la
meilleure config dans naviflow/eval/frozen_params_global.json. La config est
scorée en validation walk-forward (le test reste réservé au report final via
`make eval GLOBAL=1`).

À relancer quand on change les features du modèle global (lags, rolls), car la
config optimale dépend du jeu de features.

Usage :
    make tune_global                       # n_iter par défaut
    make tune_global N_ITER=50
"""

import os

from naviflow.ml_logic.evaluation import tune_global
from naviflow.interface.main_eval import _parse_ints


if __name__ == "__main__":
    n_iter = int(os.getenv("N_ITER", "30"))
    lags = _parse_ints(os.getenv("LAGS"), (1, 2, 3, 4, 5, 6, 7))
    rolls = _parse_ints(os.getenv("ROLLS"), (7, 14, 30))
    tune_global(n_iter=n_iter, lags=lags, rolls=rolls)
