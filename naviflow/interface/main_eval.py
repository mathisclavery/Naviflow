"""Harnais d'EVALUATION RAPIDE des experimentations XGBoost.

Entraine le modele sur l'echantillon fige de ~40 stations (avec les
hyperparametres figes, SANS recherche), le compare a la baseline naive
'meme jour de la semaine derniere', et journalise le resultat.

C'est l'outil central des ablations : on change UNE chose (une feature), on
relance `make eval`, on regarde le delta vs baseline. Tourne en moins d'une
minute, contre ~10h pour un entrainement complet 708 stations.

Usage :
    make eval LABEL="lags multi-semaines" NOTES="ajout lag_14/21/28"
    python -m naviflow.interface.main_eval
"""

import os

from tqdm import tqdm

from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.preprocess_xgb import prepare_xgb
from naviflow.ml_logic.models.baselines import run_baseline_weekday
from naviflow.ml_logic.evaluation import (
    EVAL_TRAIN_FROM,
    load_sample_stations,
    load_frozen_params,
    fit_xgb_frozen,
    baseline_metrics,
    aggregate_metrics,
    log_experiment,
)
from naviflow.utils import display as d


def run_eval(label, notes="", horizon=7, lags=(1, 7, 30),
             train_from=EVAL_TRAIN_FROM, do_log=True):
    """Evalue le modele courant sur l'echantillon fige et le compare a la baseline.

    Parametres
    ----------
    label : nom court de l'experience (pour le journal).
    notes : description libre de ce qui a change.
    horizon, lags : config de preprocessing (a faire varier lors des ablations).
    train_from : fenetre de donnees (post-COVID par defaut).
    do_log : si True, ecrit une ligne dans experiments/log.csv.
    """
    station_ids = load_sample_stations()
    params = load_frozen_params()

    d.title(f"EVAL — {label}")
    d.info(f"{len(station_ids)} stations | lags={lags} | horizon={horizon} | depuis {train_from}")

    d.step("Chargement des donnees + feature engineering")
    df = get_data(train_from=train_from)
    df = build_features(df, with_cluster=False)

    model_results = {}
    baseline_res = {}

    pbar = tqdm(station_ids, desc="Eval", unit="station")
    for gid in pbar:
        pbar.set_postfix_str(f"station {gid}")
        df_group = df[df["ID_LIEU"] == gid]
        if len(df_group) <= max(lags) + horizon + 1:
            continue

        X_np, Y_np, _, dates_np = prepare_xgb(df_group, lags=lags, horizon=horizon,
                                              as_numpy=True)
        model_results[gid] = fit_xgb_frozen(X_np, Y_np, dates_np, params)

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
    """Affiche le tableau comparatif modele vs baseline."""
    d.success("Resultats (mediane sur les stations) :")
    d.info(f"{'':12s} {'modele':>10s} {'baseline':>10s}")
    d.info(f"{'MAE% global':12s} {model_agg['mae_pct_median']:>9.1f}% {base_agg['mae_pct_median']:>9.1f}%")
    for h in range(1, horizon + 1):
        m = model_agg.get(f"mae_pct_j{h}")
        b = base_agg.get(f"mae_pct_j{h}")
        d.info(f"{'MAE% J+'+str(h):12s} {m:>9.1f}% {b:>9.1f}%")
    d.info(f"{'R2 median':12s} {model_agg['r2_median']:>10.3f} {'—':>10s}")
    d.info(f"{'R2<0':12s} {model_agg['n_r2_neg']:>10d} {'—':>10s}")

    if delta > 0:
        d.success(f"Le modele bat la baseline de {delta:.1f} points de MAE%")
    else:
        d.warn(f"Le modele NE bat PAS la baseline ({delta:.1f} points)")


if __name__ == "__main__":
    label = os.getenv("LABEL", "reference")
    notes = os.getenv("NOTES", "")
    run_eval(label=label, notes=notes)
