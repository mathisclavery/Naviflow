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

from naviflow.config import PROJECT_ROOT, N_CLUSTERS_DEFAULT, TARGET
from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import (
    cluster_stations, build_features, build_station_profiles,
)
from naviflow.ml_logic.preprocess_xgb import prepare_xgb
from naviflow.ml_logic.models.sklearn_models import run_xgboost
from naviflow.utils import display as d

# Dossier des artefacts d'experimentation figes (commites).
EVAL_DIR = PROJECT_ROOT / "naviflow" / "eval"
SAMPLE_STATIONS_PATH = EVAL_DIR / "sample_stations.json"
FROZEN_PARAMS_PATH = EVAL_DIR / "frozen_params.json"
FROZEN_GLOBAL_PARAMS_PATH = EVAL_DIR / "frozen_params_global.json"

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

# Grille pour le modèle GLOBAL poolé. Le régime change tout : ~30k lignes
# (toutes stations empilées) au lieu de ~800 par station, donc on autorise des
# arbres bien plus profonds et un learning_rate plus élevé — la config figée
# par station (max_depth=4, lr=0.01) sous-apprend largement en poolé.
GLOBAL_PARAM_GRID = {
    "n_estimators":     [300, 500, 800, 1200],
    "max_depth":        [6, 8, 10, 12],
    "learning_rate":    [0.02, 0.03, 0.05, 0.08, 0.1],
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
def _per_station_metrics(y_true, y_pred):
    """Calcule les métriques d'une station à partir de ses vraies valeurs et prédictions.

    Format partagé par TOUTES les voies d'évaluation (modèle par station, modèle
    global, baseline) pour que les chiffres soient directement comparables :
      - mae, r2          : globaux (toutes sorties confondues) ;
      - mae_pct          : MAE relative = MAE / moyenne réelle, en % ;
      - *_per_h          : détail par horizon J+1 … J+H.
    `mae_pct` vaut None si la moyenne réelle est nulle (station sans trafic).
    """
    mae_raw = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    r2_raw = r2_score(y_true, y_pred, multioutput="raw_values")
    true_mean = y_true.mean(axis=0)
    horizons = list(range(1, y_true.shape[1] + 1))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae_pct": float(mae_raw.mean() / y_true.mean() * 100) if y_true.mean() else None,
        "mae_per_h": {h: float(m) for h, m in zip(horizons, mae_raw)},
        "r2_per_h": {h: float(r) for h, r in zip(horizons, r2_raw)},
        "mae_pct_per_h": {h: float(m / mu * 100) if mu else None
                          for h, m, mu in zip(horizons, mae_raw, true_mean)},
    }


def fit_xgb_frozen(X, Y, dates, params, test_size=0.2, random_state=67,
                   log_target=False):
    """Entraîne UN XGBoost multi-sortie par station avec des hyperparamètres FIGÉS.

    Reproduit EXACTEMENT le split temporel de `run_xgboost` (même cutoff sur les
    jours uniques) pour que les métriques soient comparables à la prod, mais
    saute le RandomizedSearchCV : entraînement direct, donc rapide. C'est la
    brique des ablations — on ne tune pas, on mesure l'effet d'une feature.

    log_target : si True, entraîne sur log1p(Y) et repasse les prédictions en
    expm1 AVANT de calculer les métriques (qui restent donc en échelle originale,
    comparables à la baseline). Ablation testée et écartée : dégrade la MAE.

    Renvoie le dict de métriques de `_per_station_metrics`.
    """
    dates = np.asarray(dates, dtype="datetime64[ns]")
    unique_days = np.unique(dates)
    cutoff = unique_days[int(len(unique_days) * (1 - test_size))]

    train_mask = dates < cutoff
    X, Y = np.asarray(X), np.asarray(Y)
    train_order = np.argsort(dates[train_mask], kind="stable")

    X_train = X[train_mask][train_order]
    y_train = Y[train_mask][train_order]
    X_test = X[~train_mask]
    y_test = Y[~train_mask]

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        multi_strategy="multi_output_tree",
        eval_metric="mae",
        random_state=random_state,
        **params,
    )
    if log_target:
        model.fit(X_train, np.log1p(y_train))
        y_pred = np.expm1(model.predict(X_test))
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    return _per_station_metrics(y_test, y_pred)


def _build_pooled_matrix(df, station_ids, level_cutoff, horizon=7, lags=(1, 7, 30),
                         rolls=(), normalize=True, add_profiles=True):
    """Empile les lignes de toutes les stations en UN jeu (X, Y) poolé et normalisé.

    `level_cutoff` : seuls les jours STRICTEMENT antérieurs servent à calculer le
    niveau de chaque station (moyenne de la cible) et ses profils — garantit que
    la normalisation ne fuit pas d'info future (ni du test, ni de la validation).

    Renvoie (X, Y_abs, gids, dates, feature_names, lvl) :
      X         : features poolées, déjà normalisées (lags/rolls ÷ niveau) si demandé.
      Y_abs     : cible EN ABSOLU (non normalisée) — pour des métriques en absolu.
      gids      : id station de chaque ligne (pour les métriques par station).
      dates     : JOUR de chaque ligne (pour le split temporel).
      lvl       : niveau station de chaque ligne (pour (dé)normaliser la cible).
    """
    train_df = df[pd.to_datetime(df["JOUR"]) < level_cutoff]
    levels = train_df.groupby("ID_LIEU")[TARGET].mean()

    prof_cols = ["log_vald", "cv", "ratio_we_sem",
                 "ratio_vac_horsvac", "creux_estival"]
    profiles = (build_station_profiles(train_df).set_index("ID_LIEU")[prof_cols]
                if add_profiles else None)

    # Pré-groupage : une seule passe au lieu d'un scan complet par station.
    groups = dict(tuple(df.groupby("ID_LIEU", sort=False)))

    X_parts, Y_parts, gid_parts, date_parts, lvl_parts = [], [], [], [], []
    feature_names = None
    min_hist = max(list(lags) + list(rolls) + [1])

    for gid in station_ids:
        if gid not in levels.index or levels[gid] <= 0:
            continue
        if add_profiles and gid not in profiles.index:
            continue
        df_group = groups.get(gid)
        if df_group is None or len(df_group) <= min_hist + horizon + 1:
            continue

        X_np, Y_np, names, dates_np = prepare_xgb(
            df_group, lags=lags, rolls=rolls, horizon=horizon, as_numpy=True)

        if add_profiles:
            prof_row = profiles.loc[gid].to_numpy(dtype=float)
            prof_block = np.tile(prof_row, (len(X_np), 1))
            X_np = np.hstack([X_np, prof_block])
            if feature_names is None:
                feature_names = list(names) + prof_cols
        elif feature_names is None:
            feature_names = list(names)

        X_parts.append(X_np)
        Y_parts.append(Y_np)
        gid_parts.append(np.full(len(X_np), gid))
        date_parts.append(dates_np)
        lvl_parts.append(np.full(len(X_np), levels[gid]))

    X = np.vstack(X_parts).astype(float)
    Y_abs = np.vstack(Y_parts).astype(float)
    gids = np.concatenate(gid_parts)
    dates = np.concatenate(date_parts).astype("datetime64[ns]")
    lvl = np.concatenate(lvl_parts)

    if normalize:
        lag_idx = [i for i, n in enumerate(feature_names)
                   if n.startswith("lag_") or n.startswith("roll_")]
        if lag_idx:
            X[:, lag_idx] = X[:, lag_idx] / lvl[:, None]

    return X, Y_abs, gids, dates, feature_names, lvl


def _fit_predict_pooled(X, Y_fit, train_mask, eval_mask, params, lvl, normalize,
                        random_state=67, multi_strategy="multi_output_tree"):
    """Entraîne UN XGBoost poolé sur train_mask, prédit eval_mask, repasse en absolu."""
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        multi_strategy=multi_strategy,
        eval_metric="mae",
        random_state=random_state,
        **params,
    )
    model.fit(X[train_mask], Y_fit[train_mask])
    pred = model.predict(X[eval_mask])
    if normalize:
        pred = pred * lvl[eval_mask][:, None]
    return pred


def fit_xgb_global(df, station_ids, params, horizon=7, lags=(1, 7, 30), rolls=(),
                   test_size=0.2, random_state=67, normalize=True,
                   add_profiles=True, multi_strategy="multi_output_tree"):
    """Entraîne UN SEUL XGBoost sur TOUTES les stations poolées (approche panel).

    Contrairement à fit_xgb_frozen (un modèle par station, ~800 points chacun),
    on empile ici les lignes des `station_ids` en un seul jeu d'entraînement
    (~800 × N stations) et on entraîne UN modèle. Il apprend la *forme* partagée
    (jour de semaine, saison, météo, vacances) sur toutes les stations à la fois,
    ce qui lève le sous-apprentissage des séries individuelles courtes.

    normalize : si True (recommandé), la cible et les lags/rolls sont divisés par
    le NIVEAU de la station (moyenne de la cible sur la portion TRAIN — pas de
    fuite). Indispensable : sinon la perte quadratique est écrasée par les
    grosses stations et les petites ont une MAE% catastrophique. Le modèle
    prédit alors un ratio (~1), reconverti en absolu par × niveau.

    add_profiles : si True, ajoute des features d'identité de station (log_vald,
    cv, ratios we/vacances/été) calculées sur le TRAIN, pour situer chaque
    station sur un continuum (substitut « doux » à un one-hot des 708 stations).

    Renvoie un dict {station_id: métriques} au format de `_per_station_metrics`.
    """
    unique_days = np.sort(pd.to_datetime(df["JOUR"]).unique())
    cutoff = unique_days[int(len(unique_days) * (1 - test_size))]

    X, Y_abs, gids, dates, _, lvl = _build_pooled_matrix(
        df, station_ids, level_cutoff=cutoff, horizon=horizon, lags=lags,
        rolls=rolls, normalize=normalize, add_profiles=add_profiles)

    Y_fit = Y_abs / lvl[:, None] if normalize else Y_abs
    train_mask = dates < cutoff
    test_mask = ~train_mask

    pred = _fit_predict_pooled(X, Y_fit, train_mask, test_mask, params, lvl,
                               normalize, random_state=random_state,
                               multi_strategy=multi_strategy)

    y_test_abs = Y_abs[test_mask]
    gids_test = gids[test_mask]

    results = {}
    for gid in np.unique(gids_test):
        m = gids_test == gid
        results[int(gid)] = _per_station_metrics(y_test_abs[m], pred[m])
    return results


def _walk_forward_folds(dates, test_cutoff, n_folds=3, min_train_frac=0.5):
    """Construit des folds walk-forward (fenêtre expansive) AVANT le test.

    On découpe la région pré-test (jours < test_cutoff) en : un train initial
    (les `min_train_frac` premiers jours) puis `n_folds` fenêtres de validation
    contiguës successives. Le fold k s'entraîne sur TOUT ce qui précède sa
    fenêtre (expansif) et valide sur la fenêtre k. Moyenner le score sur ces
    folds lisse les idiosyncrasies d'une période unique — indispensable car la
    MAE% des horizons lointains est très volatile d'une période à l'autre.

    Renvoie la liste des (train_mask, val_mask) et le jour servant de level_cutoff
    (début de la 1re fenêtre de validation : aucun fold ne voit le futur via le
    niveau station).
    """
    pre = np.sort(np.unique(dates[dates < test_cutoff]))
    n_pre = len(pre)
    start = int(n_pre * min_train_frac)
    bounds = np.linspace(start, n_pre, n_folds + 1).astype(int)

    level_cutoff = pre[bounds[0]]
    folds = []
    for k in range(n_folds):
        v0, v1 = pre[bounds[k]], pre[bounds[k + 1] - 1]
        train_mask = dates < v0
        val_mask = (dates >= v0) & (dates <= v1)
        folds.append((train_mask, val_mask))
    return folds, level_cutoff


def tune_global(n_iter=30, horizon=7, lags=(1, 2, 3, 4, 5, 6, 7), rolls=(7, 14, 30),
                train_from=EVAL_TRAIN_FROM, test_size=0.2, n_folds=3,
                seed=42, save=True):
    """Tune les hyperparamètres du modèle GLOBAL poolé (recherche aléatoire + walk-forward CV).

    Méthodologie (pas de fuite, sélection robuste) :
      - TEST (derniers `test_size`) : jamais touché ici — réservé au report final
        via `make eval GLOBAL=1`. On le tient juste à l'écart.
      - VALIDATION WALK-FORWARD : la région pré-test est découpée en `n_folds`
        fenêtres temporelles successives (fenêtre d'entraînement expansive). Le
        score d'une config = MOYENNE, sur les folds, de la médiane inter-stations
        de la MAE%. Une config doit donc bien généraliser sur PLUSIEURS périodes,
        pas sur un seul bloc — ce qui évite de sélectionner sur le bruit des
        horizons lointains (cf. lr=0.1 qui sur-apprenait un bloc unique).

    On tire `n_iter` configs dans GLOBAL_PARAM_GRID, on garde la meilleure
    (score moyen le plus bas) et on la fige dans frozen_params_global.json.
    """
    station_ids = load_sample_stations()
    d.title(f"TUNING GLOBAL — {len(station_ids)} stations | n_iter={n_iter} | {n_folds} folds")
    d.info(f"lags={lags} | rolls={rolls} | test_size={test_size}")

    d.step("Chargement des donnees + feature engineering")
    df = get_data(train_from=train_from)
    df = build_features(df, with_cluster=False)

    unique_days = np.sort(pd.to_datetime(df["JOUR"]).unique())
    test_cutoff = unique_days[int(len(unique_days) * (1 - test_size))]

    # Le level_cutoff doit valoir le début de la 1re fenêtre de validation, qui
    # dépend des dates du jeu poolé. Comme les dates ne dépendent pas du cutoff, on
    # fait un 1er passage léger (sans normalisation ni profils) juste pour les
    # obtenir, on en dérive les folds, puis on (re)construit le jeu avec le bon cutoff.
    d.step("Construction du jeu poolé")
    _, _, _, dates_probe, _, _ = _build_pooled_matrix(
        df, station_ids, level_cutoff=test_cutoff, horizon=horizon, lags=lags,
        rolls=rolls, normalize=False, add_profiles=False)
    folds, level_cutoff = _walk_forward_folds(dates_probe, test_cutoff, n_folds=n_folds)

    X, Y_abs, gids, dates, _, lvl = _build_pooled_matrix(
        df, station_ids, level_cutoff=level_cutoff, horizon=horizon, lags=lags,
        rolls=rolls, normalize=True, add_profiles=True)
    Y_fit = Y_abs / lvl[:, None]
    for k, (tr, va) in enumerate(folds):
        d.info(f"fold {k+1}: {tr.sum()} train | {va.sum()} val")

    # Tirage des configs.
    rng = np.random.default_rng(seed)
    grid_keys = list(GLOBAL_PARAM_GRID)
    sampled = []
    seen = set()
    while len(sampled) < n_iter:
        cfg = {k: GLOBAL_PARAM_GRID[k][rng.integers(len(GLOBAL_PARAM_GRID[k]))]
               for k in grid_keys}
        key = tuple(cfg[k] for k in grid_keys)
        if key not in seen:
            seen.add(key)
            sampled.append(cfg)

    def _fold_score(cfg, train_mask, val_mask):
        pred = _fit_predict_pooled(X, Y_fit, train_mask, val_mask, cfg, lvl,
                                   normalize=True)
        gids_val, y_val = gids[val_mask], Y_abs[val_mask]
        results = {int(g): _per_station_metrics(y_val[gids_val == g], pred[gids_val == g])
                   for g in np.unique(gids_val)}
        return aggregate_metrics(results, horizon=horizon)["mae_pct_median"]

    best = {"score": np.inf, "params": None}
    pbar = tqdm(sampled, desc="Tuning", unit="config")
    for cfg in pbar:
        fold_scores = [_fold_score(cfg, tr, va) for tr, va in folds]
        score = float(np.mean(fold_scores))
        pbar.set_postfix_str(f"MAE% {score:.2f} (best {best['score']:.2f})")
        if score < best["score"]:
            best = {"score": score, "params": cfg, "fold_scores": fold_scores}

    d.success(f"Meilleure config (MAE% moyenne walk-forward = {best['score']:.2f}) :")
    d.info(f"scores par fold : {[round(s, 2) for s in best['fold_scores']]}")
    for k, v in best["params"].items():
        d.info(f"{k:18s} = {v}")

    if save:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_on": date.today().isoformat(),
            "method": "random search on pooled global model, scored by mean (over walk-forward folds) of median per-station MAE%",
            "params": {
                "n_iter": n_iter,
                "horizon": horizon,
                "lags": list(lags),
                "rolls": list(rolls),
                "train_from": train_from,
                "test_size": test_size,
                "n_folds": n_folds,
                "n_stations": len(station_ids),
            },
            "val_mae_pct_mean": best["score"],
            "val_fold_scores": best["fold_scores"],
            "consensus_params": best["params"],
        }
        FROZEN_GLOBAL_PARAMS_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False))
        d.success(f"Params global figes : {FROZEN_GLOBAL_PARAMS_PATH.relative_to(PROJECT_ROOT)}")

    return best["params"]


def load_frozen_global_params():
    """Charge la config d'hyperparamètres figée du modèle global poolé."""
    if not FROZEN_GLOBAL_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Params global introuvables : {FROZEN_GLOBAL_PARAMS_PATH}. "
            "Génère-les d'abord avec tune_global()."
        )
    return json.loads(FROZEN_GLOBAL_PARAMS_PATH.read_text())["consensus_params"]


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
