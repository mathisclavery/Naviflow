"""Etage d'EXPERIMENTATION (séparé du pipeline de prod).

Ce module fournit l'outillage pour itérer vite sur l'amélioration du modele
XGBoost sans relancer un entrainement complet :

  1. `sample_stations()` : tire un echantillon stratifié de stations (n par
     cluster KMeans) pour evaluer sur ~40 stations representatives au lieu de
     708. Comme la recherche d'hyperparametres est désactivée pendant les
     ablations, entrainer 40 stations est rapide.

  2. (à venir) `aggregate_metrics()` et `log_experiment()` : agregation des
     metriques entre stations et journal d'experiences.

Rien ici ne touche au pipeline de prod (`main_xgb.py`, `registry_xgb.py`).
L'echantillon est fige sur disque (`naviflow/eval/sample_stations.json`) pour
que toutes les ablations soient comparees sur EXACTEMENT les memes stations.
"""

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from naviflow.config import PROJECT_ROOT, N_CLUSTERS_DEFAULT
from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import cluster_stations, build_features
from naviflow.ml_logic.preprocess_xgb import prepare_xgb
from naviflow.ml_logic.models.sklearn_models import run_xgboost
from naviflow.utils import display as d

# Dossier des artefacts d'experimentation figes (commites).
EVAL_DIR = PROJECT_ROOT / "naviflow" / "eval"
SAMPLE_STATIONS_PATH = EVAL_DIR / "sample_stations.json"
FROZEN_PARAMS_PATH = EVAL_DIR / "frozen_params.json"

# Journal d'experiences (une ligne par essai).
EXPERIMENTS_LOG = PROJECT_ROOT / "experiments" / "log.csv"

# Fenetre post-COVID par defaut pour les experimentations (donnees propres).
EVAL_TRAIN_FROM = "2023-01-01"

# Grille RESSERREE pour la passe consensus. On plafonne max_depth a 6 et
# n_estimators a 1000 : sur ~800 points par station, des arbres plus gros ne
# font que sur-apprendre (cf. les 28 stations a R2<0 du modele actuel).
RESTRICTED_PARAM_GRID = {
    "n_estimators":     [300, 500, 800, 1000],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.02, 0.03, 0.05, 0.08],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],
    "gamma":            [0, 0.1, 0.3, 0.5],
    "reg_alpha":        [0, 0.1, 0.5, 1.0],
    "reg_lambda":       [0.5, 1.0, 1.5, 2.0, 3.0],
}

# Hyperparametres a arrondir a l'entier lors de l'agregation.
_INT_PARAMS = {"n_estimators", "max_depth", "min_child_weight"}


def sample_stations(n_per_cluster=10, n_clusters=N_CLUSTERS_DEFAULT,
                    train_from=EVAL_TRAIN_FROM, seed=42, save=True):
    """Tire un echantillon STRATIFIE de stations pour l'evaluation.

    Strategie : on clusterise les stations par profil de trafic (KMeans, via
    `cluster_stations`), puis on pioche `n_per_cluster` stations dans CHAQUE
    cluster. L'echantillon couvre ainsi toutes les familles de stations (gros
    poles, stations moyennes, petites) dans les bonnes proportions — sa mediane
    est representative des 708 stations, contrairement a un tirage aleatoire qui
    pourrait ne contenir que de grosses stations.

    Parametres
    ----------
    n_per_cluster : nombre de stations a tirer par cluster.
    n_clusters : nombre de clusters KMeans.
    train_from : fenetre de donnees utilisee pour profiler/clusteriser les
        stations (defaut : post-COVID, coherent avec la fenetre d'eval).
    seed : graine du tirage — FIGEE pour la reproductibilite. Deux appels avec
        la meme graine renvoient exactement les memes stations.
    save : si True, ecrit l'echantillon dans `sample_stations.json`.

    Renvoie
    -------
    list[int] : les ID_LIEU tires, tries.
    """
    d.step(f"Tirage stratifie : {n_per_cluster} stations x {n_clusters} clusters")

    df = get_data(train_from=train_from)
    cluster_map = cluster_stations(df, n=n_clusters)  # [ID_LIEU, cluster]

    # Pioche n_per_cluster par cluster. Les clusters comptent ~150-200 stations
    # chacun, largement assez pour en tirer n_per_cluster.
    sampled = (
        cluster_map
        .groupby("cluster", group_keys=False)
        .sample(n=n_per_cluster, random_state=seed)
    )

    station_ids = sorted(int(s) for s in sampled["ID_LIEU"])

    sizes = sampled.groupby("cluster").size().to_dict()
    d.info(f"{len(station_ids)} stations tirees | par cluster : {sizes}")

    if save:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_on": date.today().isoformat(),
            "params": {
                "n_per_cluster": n_per_cluster,
                "n_clusters": n_clusters,
                "train_from": train_from,
                "seed": seed,
            },
            "station_ids": station_ids,
            "by_cluster": {
                int(c): sorted(int(s) for s in g["ID_LIEU"])
                for c, g in sampled.groupby("cluster")
            },
        }
        SAMPLE_STATIONS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        d.success(f"Echantillon fige : {SAMPLE_STATIONS_PATH.relative_to(PROJECT_ROOT)}")

    return station_ids


def load_sample_stations():
    """Charge la liste figee des stations d'evaluation.

    Leve une erreur explicite si l'echantillon n'a pas encore ete genere.
    """
    if not SAMPLE_STATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Echantillon introuvable : {SAMPLE_STATIONS_PATH}. "
            "Genere-le d'abord avec sample_stations()."
        )
    payload = json.loads(SAMPLE_STATIONS_PATH.read_text())
    return payload["station_ids"]


def _aggregate_params(params_list):
    """Agrege une liste de best_params en une config consensus (mediane param par param).

    La mediane est robuste : un coup de chance du tuning sur une station isolee
    ne tire pas la config consensus. Les params entiers sont arrondis.
    """
    keys = sorted({k for p in params_list for k in p})
    consensus = {}
    for k in keys:
        vals = [p[k] for p in params_list if k in p]
        med = float(np.median(vals))
        consensus[k] = int(round(med)) if k in _INT_PARAMS else round(med, 4)
    return consensus


def consensus_tuning(n_iter=15, horizon=7, lags=(1, 7, 30),
                     train_from=EVAL_TRAIN_FROM, save=True):
    """Derive UNE config d'hyperparametres figee depuis l'echantillon de stations.

    Pour chaque station de l'echantillon, lance un RandomizedSearchCV (grille
    resserree) et recupere ses best_params. La config consensus = mediane de
    chaque hyperparametre sur les 40 stations. On la fige ensuite et on l'utilise
    pour TOUTES les ablations de features : ainsi un gain mesure vient de la
    feature, pas d'un coup de chance du tuning.

    A ne lancer qu'une fois (setup). Le tuning fin du modele final, lui, se fera
    a la fin sur les features retenues.
    """
    station_ids = load_sample_stations()
    d.title(f"TUNING CONSENSUS — {len(station_ids)} stations | n_iter={n_iter}")

    d.step("Chargement des donnees + feature engineering")
    df = get_data(train_from=train_from)
    df = build_features(df, with_cluster=False)

    best_params_all = {}
    pbar = tqdm(station_ids, desc="Tuning", unit="station")
    for gid in pbar:
        pbar.set_postfix_str(f"station {gid}")
        df_group = df[df["ID_LIEU"] == gid]
        if len(df_group) <= max(lags) + horizon + 1:
            continue

        X_np, Y_np, _, dates_np = prepare_xgb(df_group, lags=lags, horizon=horizon,
                                              as_numpy=True)
        res = run_xgboost(X_np, Y_np, dates_np, n_iter=n_iter,
                          param_grid=RESTRICTED_PARAM_GRID)
        best_params_all[int(gid)] = res["best_params"]

    consensus = _aggregate_params(list(best_params_all.values()))

    d.success("Config consensus (mediane sur les stations) :")
    for k, v in consensus.items():
        d.info(f"{k:18s} = {v}")

    if save:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_on": date.today().isoformat(),
            "method": "median of per-station RandomizedSearchCV best_params",
            "params": {
                "n_iter": n_iter,
                "horizon": horizon,
                "lags": list(lags),
                "train_from": train_from,
                "n_stations": len(best_params_all),
            },
            "consensus_params": consensus,
            "per_station_best_params": best_params_all,
        }
        FROZEN_PARAMS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        d.success(f"Params figes : {FROZEN_PARAMS_PATH.relative_to(PROJECT_ROOT)}")

    return consensus


def load_frozen_params():
    """Charge la config d'hyperparametres figee (consensus_params)."""
    if not FROZEN_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Params figes introuvables : {FROZEN_PARAMS_PATH}. "
            "Genere-les d'abord avec consensus_tuning()."
        )
    return json.loads(FROZEN_PARAMS_PATH.read_text())["consensus_params"]


# --------------------------------------------------------------------------- #
# Entrainement DIRECT (sans recherche) — pour les ablations
# --------------------------------------------------------------------------- #
def fit_xgb_frozen(X, Y, dates, params, test_size=0.2, random_state=67):
    """Entraine UN XGBoost multi-sortie avec des hyperparametres FIGES (sans search).

    Reproduit EXACTEMENT le split temporel de `run_xgboost` (meme cutoff sur les
    jours uniques) pour que les metriques soient comparables a la prod, mais
    saute le RandomizedSearchCV : entrainement direct, donc rapide. C'est la
    brique des ablations (on ne tune pas, on mesure l'effet d'une feature).

    Renvoie un dict de metriques par station : mae, r2, mae_pct (relatif),
    mae_per_h, r2_per_h, mae_pct_per_h.
    """
    dates = np.asarray(dates, dtype="datetime64[ns]")
    unique_days = np.unique(dates)
    cutoff = unique_days[int(len(unique_days) * (1 - test_size))]

    train_mask = dates < cutoff
    test_mask = ~train_mask

    X = np.asarray(X)
    Y = np.asarray(Y)
    train_order = np.argsort(dates[train_mask], kind="stable")

    X_train = X[train_mask][train_order]
    y_train = Y[train_mask][train_order]
    X_test = X[test_mask]
    y_test = Y[test_mask]

    base_kwargs = dict(
        objective="reg:squarederror",
        tree_method="hist",
        multi_strategy="multi_output_tree",
        eval_metric="mae",
        random_state=random_state,
    )
    model = XGBRegressor(**base_kwargs, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mae_raw = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    r2_raw = r2_score(y_test, y_pred, multioutput="raw_values")
    true_mean = y_test.mean(axis=0)
    horizons = list(range(1, Y.shape[1] + 1))

    return {
        "mae": float(mae),
        "r2": float(r2),
        "mae_pct": float(mae / y_test.mean() * 100),
        "mae_per_h": {h: float(m) for h, m in zip(horizons, mae_raw)},
        "r2_per_h": {h: float(r) for h, r in zip(horizons, r2_raw)},
        "mae_pct_per_h": {h: float(m / mu * 100) if mu else None
                          for h, m, mu in zip(horizons, mae_raw, true_mean)},
    }


def baseline_metrics(b, horizon=7):
    """Normalise la sortie de run_baseline_weekday au meme format que fit_xgb_frozen.

    La baseline n'a pas de R2 (ce n'est pas un modele appris) -> r2=None.
    mae_pct global = MAE / moyenne reelle du test ; mae_pct_per_h vient de 'flat'.
    """
    y_test = b["y_test"]
    return {
        "mae_pct": float(b["mae"] / y_test.mean() * 100),
        "r2": None,
        "mae_pct_per_h": {h: b["flat"].get(f"pct_j{h}") for h in range(1, horizon + 1)},
    }


# --------------------------------------------------------------------------- #
# Agregation entre stations + journal d'experiences
# --------------------------------------------------------------------------- #
def aggregate_metrics(results, horizon=7):
    """Agrege les metriques par station en MEDIANES (robustes aux outliers).

    `results` : dict {station_id: metriques} produit par fit_xgb_frozen (ou un
    dict equivalent pour la baseline, sans 'r2'/'r2_per_h'). On prend la mediane
    entre stations car la moyenne est polluee par les stations a R2<0.

    Renvoie un dict agrege : mae_pct_median (global + par horizon), r2_median,
    n_stations, n_r2_neg.
    """
    vals = list(results.values())
    horizons = list(range(1, horizon + 1))

    mae_pct = [v["mae_pct"] for v in vals if v.get("mae_pct") is not None]
    r2 = [v["r2"] for v in vals if v.get("r2") is not None]

    agg = {
        "n_stations": len(vals),
        "mae_pct_median": round(float(np.median(mae_pct)), 2) if mae_pct else None,
        "r2_median": round(float(np.median(r2)), 4) if r2 else None,
        "n_r2_neg": int(sum(1 for x in r2 if x < 0)) if r2 else None,
    }
    for h in horizons:
        col = [v["mae_pct_per_h"][h] for v in vals
               if v.get("mae_pct_per_h", {}).get(h) is not None]
        agg[f"mae_pct_j{h}"] = round(float(np.median(col)), 2) if col else None

    return agg


def log_experiment(label, agg, notes="", delta_vs_baseline=None, horizon=7):
    """Ajoute une ligne au journal d'experiences (experiments/log.csv).

    Cree le fichier (avec en-tete) au premier appel. Une ligne = un essai :
    horodatage, label, metriques agregees, delta vs baseline, notes.
    """
    EXPERIMENTS_LOG.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "mae_pct_median": agg.get("mae_pct_median"),
        "r2_median": agg.get("r2_median"),
        "delta_vs_baseline_pct": delta_vs_baseline,
        "n_r2_neg": agg.get("n_r2_neg"),
        "n_stations": agg.get("n_stations"),
    }
    for h in range(1, horizon + 1):
        row[f"mae_pct_j{h}"] = agg.get(f"mae_pct_j{h}")
    row["notes"] = notes

    df_row = pd.DataFrame([row])
    header = not EXPERIMENTS_LOG.exists()
    df_row.to_csv(EXPERIMENTS_LOG, mode="a", header=header, index=False)
    d.success(f"Experience journalisee : {EXPERIMENTS_LOG.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    d.title("TIRAGE DE L'ECHANTILLON D'EVALUATION")
    ids = sample_stations()
    d.done(f"{len(ids)} stations pretes pour les experimentations")
