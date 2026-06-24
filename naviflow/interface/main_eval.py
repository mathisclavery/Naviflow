"""Harnais d'ÉVALUATION RAPIDE des expérimentations XGBoost.

Entraîne un modèle sur l'échantillon figé de ~40 stations (hyperparamètres figés,
SANS recherche), le compare à la baseline naïve « même jour de la semaine
dernière » et journalise le résultat dans experiments/log.csv.

C'est l'outil central des ablations : on change UNE chose (une feature, le mode
de modèle…), on relance `make eval`, on regarde le delta vs baseline.

Deux modes de modèle :
  - par station (défaut) : un XGBoost par station, rapide sur l'échantillon ;
  - global poolé (`GLOBAL=1`) : UN modèle sur toutes les stations empilées.

Usage :
    make eval LABEL="rolling" ROLLS="7,14,30"
    make eval LABEL="global" GLOBAL=1 LAGS="1,2,3,4,5,6,7" ROLLS="7,14,30"
    make eval LABEL="global 697" GLOBAL=1 ALL=1        # confirmation grande échelle
"""

import os

from tqdm import tqdm

from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.preprocess_xgb import prepare_xgb
from naviflow.ml_logic.models.baselines import run_baseline_weekday
from naviflow.ml_logic.evaluation import (
    EVAL_TRAIN_FROM,
    FROZEN_GLOBAL_PARAMS_PATH,
    load_sample_stations,
    load_frozen_params,
    load_frozen_global_params,
    fit_xgb_frozen,
    fit_xgb_global,
    baseline_metrics,
    aggregate_metrics,
    log_experiment,
)
from naviflow.utils import display as d


def run_eval(label, notes="", horizon=7, lags=(1, 7, 30), rolls=(),
             log_target=False, model_global=False, all_stations=False,
             train_from=EVAL_TRAIN_FROM, do_log=True):
    """Évalue un modèle sur l'échantillon (ou toutes les stations) vs la baseline.

    Paramètres
    ----------
    label, notes : nom court et description libre de l'expérience (pour le journal).
    horizon, lags, rolls : config de preprocessing (à faire varier en ablation).
    log_target : entraîne sur log1p(cible) — uniquement en mode par station.
    model_global : si True, entraîne UN SEUL modèle poolé sur toutes les stations
        (approche panel, cible normalisée par niveau station) au lieu d'un modèle
        par station.
    all_stations : si True, évalue sur TOUTES les stations du dataset (et non
        l'échantillon figé de 40) — pour confirmer un gain à grande échelle.
        À réserver au mode global (un seul fit), sinon très long.
    train_from : fenêtre de données (post-COVID par défaut).
    do_log : si True, écrit une ligne dans experiments/log.csv.
    """
    # En mode global on préfère les params tunés pour le régime poolé (s'ils
    # existent) ; sinon on retombe sur les params par station (sous-dimensionnés).
    if model_global and FROZEN_GLOBAL_PARAMS_PATH.exists():
        params = load_frozen_global_params()
    else:
        params = load_frozen_params()

    d.title(f"EVAL — {label}")

    d.step("Chargement des données + feature engineering")
    df = get_data(train_from=train_from)
    df = build_features(df, with_cluster=False)

    # Échantillon figé (40 stations) ou toutes les stations du dataset.
    station_ids = (sorted(int(s) for s in df["ID_LIEU"].unique())
                   if all_stations else load_sample_stations())

    mode = "GLOBAL poolé" if model_global else "par station"
    d.info(f"{len(station_ids)} stations | mode={mode} | lags={lags} | rolls={rolls} | horizon={horizon} | depuis {train_from}")

    model_results = {}
    baseline_res = {}

    # Le modèle global est UN seul fit sur toutes les stations empilées.
    if model_global:
        d.step("Entraînement du modèle global poolé")
        model_results = fit_xgb_global(df, station_ids, params, horizon=horizon,
                                        lags=lags, rolls=rolls)

    # Boucle par station : baseline pour tous, + le modèle par station si non global.
    groups = dict(tuple(df.groupby("ID_LIEU", sort=False)))
    min_hist = max(list(lags) + list(rolls) + [1])
    for gid in tqdm(station_ids, desc="Eval", unit="station"):
        df_group = groups.get(gid)
        if df_group is None or len(df_group) <= min_hist + horizon + 1:
            continue

        if not model_global:
            X_np, Y_np, _, dates_np = prepare_xgb(df_group, lags=lags, rolls=rolls,
                                                  horizon=horizon, as_numpy=True)
            model_results[gid] = fit_xgb_frozen(X_np, Y_np, dates_np, params,
                                                log_target=log_target)

        b = run_baseline_weekday(df_group, horizon=horizon, lags=lags)
        baseline_res[gid] = baseline_metrics(b, horizon=horizon)

    model_agg = aggregate_metrics(model_results, horizon=horizon)
    base_agg = aggregate_metrics(baseline_res, horizon=horizon)

    delta = round(base_agg["mae_pct_median"] - model_agg["mae_pct_median"], 2)

    _print_comparison(model_agg, base_agg, delta, horizon)

    if do_log:
        log_experiment(label, model_agg, notes=notes,
                       delta_vs_baseline=delta, horizon=horizon)

    return model_agg, base_agg, delta


def _print_comparison(model_agg, base_agg, delta, horizon):
    """Affiche le tableau comparatif modèle vs baseline."""
    d.success("Résultats (médiane sur les stations) :")
    d.info(f"{'':12s} {'modèle':>10s} {'baseline':>10s}")
    d.info(f"{'MAE% global':12s} {model_agg['mae_pct_median']:>9.1f}% {base_agg['mae_pct_median']:>9.1f}%")
    for h in range(1, horizon + 1):
        m = model_agg.get(f"mae_pct_j{h}")
        b = base_agg.get(f"mae_pct_j{h}")
        d.info(f"{'MAE% J+'+str(h):12s} {m:>9.1f}% {b:>9.1f}%")
    d.info(f"{'R2 médian':12s} {model_agg['r2_median']:>10.3f} {'—':>10s}")
    d.info(f"{'R2<0':12s} {model_agg['n_r2_neg']:>10d} {'—':>10s}")

    if delta > 0:
        d.success(f"Le modèle bat la baseline de {delta:.1f} points de MAE%")
    else:
        d.warn(f"Le modèle NE bat PAS la baseline ({delta:.1f} points)")


def _parse_ints(env_val, default):
    """Parse une liste d'entiers depuis une variable d'env ('1,7,14') -> (1, 7, 14)."""
    if not env_val:
        return default
    return tuple(int(x) for x in env_val.split(",") if x.strip())


if __name__ == "__main__":
    label = os.getenv("LABEL", "reference")
    notes = os.getenv("NOTES", "")
    lags = _parse_ints(os.getenv("LAGS"), (1, 7, 30))
    rolls = _parse_ints(os.getenv("ROLLS"), ())
    log_target = os.getenv("LOG_TARGET", "").lower() in ("1", "true", "yes")
    model_global = os.getenv("GLOBAL", "").lower() in ("1", "true", "yes")
    all_stations = os.getenv("ALL", "").lower() in ("1", "true", "yes")
    run_eval(label=label, notes=notes, lags=lags, rolls=rolls,
             log_target=log_target, model_global=model_global,
             all_stations=all_stations)
