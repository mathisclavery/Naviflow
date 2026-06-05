import gc
import os

from tqdm import tqdm

from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.preprocess_xgb import prepare_xgb
from naviflow.ml_logic.models.sklearn_models import run_xgboost
from naviflow import registry_xgb
from naviflow.utils import display as d
from naviflow.config import TRAIN_FROM as DEFAULT_TRAIN_FROM


def train_all(grain="station", n_clusters=4, lags=(1, 7, 30), horizon=7,
              n_iter=40, save=True, force=False):
    """Entraine UN modele multi-sortie par groupe, predisant J+1..J+horizon."""

    actual_train_from = os.getenv("TRAIN_FROM", DEFAULT_TRAIN_FROM)
    d.title(f"PIPELINE XGBOOST — grain={grain.upper()} | sortie=J+1..J+{horizon}")

    d.step("Chargement des donnees (get_data)")
    with_cluster = (grain == "cluster")
    df = get_data()
    d.info(f"{len(df):,} lignes | periode {df['JOUR'].min():%Y-%m-%d} -> {df['JOUR'].max():%Y-%m-%d}")

    d.step("Feature engineering (build_features)")
    df = build_features(df, with_cluster=with_cluster, n_clusters=n_clusters)
    d.info(f"{df.shape[1]} colonnes apres enrichissement")

    group_col = "ID_LIEU" if grain == "station" else "cluster"
    group_ids = sorted(g for g in df[group_col].dropna().unique())
    d.step(f"Entrainement de {len(group_ids)} modeles ({grain})")

    results = {}
    skipped = 0

    pbar = tqdm(group_ids, desc="Entrainement", unit="modele")
    for gid in pbar:
        pbar.set_postfix_str(f"{grain} {gid}")

        if not force and save and registry_xgb.model_path(gid, grain, horizon, actual_train_from).exists():
            skipped += 1
            continue

        df_group = df[df[group_col] == gid]
        if len(df_group) <= max(lags) + horizon + 1:
            continue

        X_np, Y_np, _, dates_np = prepare_xgb(df_group, lags=lags, horizon=horizon, as_numpy=True)
        res = run_xgboost(X_np, Y_np, dates_np, n_iter=n_iter)

        mae_pct = res["mae"] / res["y_test"].mean() * 100
        results[gid] = {"mae": res["mae"], "r2": res["r2"], "mae_cv": res["mae_cv"],
                        "mae_per_h": res["mae_per_h"], "r2_per_h": res["r2_per_h"],
                        "mae_pct": round(mae_pct, 1), "n_samples": len(Y_np)}

        if save:
            registry_xgb.save_model(res["model"], gid, grain=grain, horizon=horizon,
                                train_from=actual_train_from)

        del df_group, X_np, Y_np, dates_np, res
        gc.collect()

    if skipped:
        d.info(f"{skipped} modeles deja existants — ignores (FORCE=1 pour reentrainer)")
    if save and results:
        path = registry_xgb.save_results(results, grain=grain, horizon=horizon,
                                     train_from=actual_train_from)
        d.success(f"Metriques : {path}")
        d.success(f"{len(results)} modeles dans : {registry_xgb.run_dir(grain, horizon, actual_train_from)}")

    d.done(f"Pipeline terminee ({len(results)} entraines, {skipped} ignores)")
    return results


if __name__ == "__main__":
    grain   = os.getenv("GRAIN", "station")
    n_iter  = int(os.getenv("N_ITER", "40"))
    horizon = os.getenv("HORIZON")
    horizon = int(horizon) if horizon else 7
    force   = os.getenv("FORCE", "0") == "1"

    train_all(grain=grain, n_iter=n_iter, horizon=horizon, force=force)
