import gc
import os

from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.preprocess_xgb import prepare_xgb
from naviflow.ml_logic.models.sklearn_models import run_xgboost
from naviflow import registry
from naviflow.utils import display as d


def train_all(grain="station", n_clusters=4, lags=(1, 7, 30), horizon=None,
              n_iter=50, save=True, force=False):

    d.title(f"PIPELINE XGBOOST — grain = {grain.upper()}")

    d.step("Chargement des donnees (get_data)")
    with_cluster = (grain == "cluster")
    df = get_data()
    d.info(f"{len(df):,} lignes chargees")

    d.step("Feature engineering (build_features)")
    df = build_features(df, with_cluster=with_cluster, n_clusters=n_clusters)
    d.info(f"{df.shape[1]} colonnes apres enrichissement")

    group_col = "ID_LIEU" if grain == "station" else "cluster"
    group_ids = sorted(g for g in df[group_col].dropna().unique())
    d.step(f"Entrainement de {len(group_ids)} modeles ({grain})")
    print()

    results = {}

    for i, gid in enumerate(group_ids, 1):

        # Skip seulement si on ne force PAS et que le modèle existe
        if not force and registry.model_path(gid, grain).exists():
            d.info(f"[{i}/{len(group_ids)}] {grain} {gid} deja fait — skip")
            continue

        df_group = df[df[group_col] == gid]

        if len(df_group) <= max(lags) + 1:
            d.warn(f"[{i}/{len(group_ids)}] {grain} {gid} ignore (trop peu de donnees)")
            continue

        X_np, y_np, _ = prepare_xgb(df_group, lags=lags, horizon=horizon, as_numpy=True)
        res = run_xgboost(X_np, y_np, n_iter=n_iter)

        results[gid] = {"mae": res["mae"], "r2": res["r2"],
                        "mae_cv": res["mae_cv"], "n_samples": len(y_np)}

        if save:
            registry.save_model(res["model"], gid, grain=grain)

        d.progress(i, len(group_ids),
                   f"{grain} {gid} — MAE={res['mae']:.0f}  R2={res['r2']:.3f}  (n={len(y_np):,})")

        del df_group, X_np, y_np, res
        gc.collect()

    if save and results:
        registry.save_results(results, grain=grain)
        d.success(f"Metriques sauvegardees : {registry.RESULTS_CSV}")
        d.success(f"{len(results)} modeles sauvegardes dans : {registry.MODELS_STORE}")

    d.done(f"Pipeline XGBoost terminee ({len(results)} modeles entraines)")
    return results


if __name__ == "__main__":
    grain   = os.getenv("GRAIN", "station")
    n_iter  = int(os.getenv("N_ITER", "50"))
    horizon = os.getenv("HORIZON")
    horizon = int(horizon) if horizon else None
    force = os.gentenv("FORCE", "0") == "1"

    train_all(grain=grain, n_iter=n_iter, horizon=horizon, force=force)
