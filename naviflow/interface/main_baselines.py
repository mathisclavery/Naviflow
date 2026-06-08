import os

import numpy as np
from tqdm import tqdm

from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.models.baselines import run_baseline_weekday
from naviflow import registry_xgb
from naviflow.utils import display as d
from naviflow.config import TRAIN_FROM as DEFAULT_TRAIN_FROM


def run_all(grain="station", n_clusters=4, lags=(1, 7, 30), horizon=7,
            test_size=0.2, save=True):
    """Baseline 'meme jour semaine derniere' par groupe, CSV propre par horizon."""

    actual_train_from = os.getenv("TRAIN_FROM", DEFAULT_TRAIN_FROM)
    d.title(f"BASELINE — grain={grain.upper()} | sortie=J+1..J+{horizon}")

    d.step("Chargement des donnees (get_data)")
    with_cluster = (grain == "cluster")
    df = get_data()
    d.info(f"{len(df):,} lignes | periode {df['JOUR'].min():%Y-%m-%d} -> {df['JOUR'].max():%Y-%m-%d}")

    d.step("Feature engineering (build_features)")
    df = build_features(df, with_cluster=with_cluster, n_clusters=n_clusters)

    group_col = "ID_LIEU" if grain == "station" else "cluster"
    group_ids = sorted(g for g in df[group_col].dropna().unique())
    d.step(f"Calcul des baselines sur {len(group_ids)} groupes ({grain})")

    results = {}
    pbar = tqdm(group_ids, desc="Baselines", unit="groupe")
    for gid in pbar:
        pbar.set_postfix_str(f"{grain} {gid}")

        df_group = df[df[group_col] == gid]
        if len(df_group) <= max(lags) + horizon + 1:
            continue

        res = run_baseline_weekday(df_group, horizon=horizon, lags=lags,
                                   test_size=test_size)
        results[gid] = res["flat"]

    if save and results:
        path = registry_xgb.save_results(results, grain=grain, horizon=horizon,
                                     train_from=actual_train_from,
                                     suffix="baseline")
        d.success(f"Metriques baseline : {path}")

    if results:
        mean_mae = np.mean([v["mae_j1"] for v in results.values()])
        d.info(f"MAE J+1 moyen (tous {grain}s) : {mean_mae:.0f}")

    d.done(f"Baseline terminee ({len(results)} groupes)")
    return results


if __name__ == "__main__":
    grain   = os.getenv("GRAIN", "station")
    horizon = os.getenv("HORIZON")
    horizon = int(horizon) if horizon else 7

    run_all(grain=grain, horizon=horizon)
