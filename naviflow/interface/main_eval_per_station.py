"""Évaluation PAR STATION du modèle global déployé, sur la période held-out.

Charge le bundle déjà entraîné (pas de réentraînement), prédit sur la fenêtre
held-out (≥ DEPLOY_TEST_CUTOFF, jamais vue) et compare au réel, station par
station. Écrit un CSV trié par % d'erreur croissant et affiche un résumé.

Usage :
    make eval_per_station
    python -m naviflow.interface.main_eval_per_station
"""

import numpy as np
import pandas as pd

from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.global_model import build_pooled_matrix
from naviflow.ml_logic.evaluation import _per_station_metrics
from naviflow import registry_xgb
from naviflow.utils import display as d
from naviflow.config import TRAIN_FROM, EXCLUDE_WINDOW, DEPLOY_TEST_CUTOFF, PROJECT_ROOT

OUT_PATH = PROJECT_ROOT / "experiments" / "results_per_station_global.csv"


def run(train_from=TRAIN_FROM, exclude_window=EXCLUDE_WINDOW,
        test_cutoff_date=DEPLOY_TEST_CUTOFF, save=True):
    """Métriques par station sur le held-out, à partir du bundle déployé."""
    d.title("ÉVALUATION PAR STATION — MODÈLE GLOBAL (held-out)")

    bundle = registry_xgb.load_global_bundle(train_from=train_from)
    model = bundle["model"]
    lags = tuple(bundle["meta"]["lags"])
    rolls = tuple(bundle["meta"]["rolls"])
    horizon = bundle["meta"]["horizon"]
    d.info(f"bundle chargé | lags={lags} | rolls={rolls} | cutoff={test_cutoff_date}")

    d.step("Chargement des données + feature engineering")
    df = get_data(train_from=train_from)
    df = build_features(df, with_cluster=False)
    station_ids = sorted(int(s) for s in df["ID_LIEU"].unique())
    labels = (df.dropna(subset=["LIBELLE_ARRET"])
              .groupby("ID_LIEU")["LIBELLE_ARRET"].first().to_dict()
              if "LIBELLE_ARRET" in df.columns else {})

    d.step("Construction des features + prédiction held-out")
    cutoff = np.datetime64(pd.Timestamp(test_cutoff_date))
    X, Y_abs, gids, dates, _, lvl = build_pooled_matrix(
        df, station_ids, level_cutoff=pd.Timestamp(test_cutoff_date), horizon=horizon,
        lags=lags, rolls=rolls, normalize=True, add_profiles=True,
        exclude_window=exclude_window)

    mask = dates >= cutoff
    pred = model.predict(X[mask]) * lvl[mask][:, None]   # ratio -> absolu
    y_true = Y_abs[mask]
    gids_test = gids[mask]

    d.step("Métriques par station")
    rows = []
    for gid in np.unique(gids_test):
        m = gids_test == gid
        if m.sum() == 0:
            continue
        met = _per_station_metrics(y_true[m], pred[m])
        rows.append({
            "station_id": int(gid),
            "libelle": labels.get(int(gid), ""),
            "n_test": int(m.sum()),
            "mae": round(met["mae"], 1),
            "mae_pct": round(met["mae_pct"], 2) if met["mae_pct"] is not None else None,
            "r2": round(met["r2"], 4),
            **{f"mae_pct_j{h}": round(met["mae_pct_per_h"][h], 2)
               for h in range(1, horizon + 1) if met["mae_pct_per_h"].get(h) is not None},
        })

    res = pd.DataFrame(rows).sort_values("mae_pct").reset_index(drop=True)

    med = res["mae_pct"].median()
    n_neg = int((res["r2"] < 0).sum())
    d.success(f"{len(res)} stations | MAE% médian = {med:.2f}% | R²<0 : {n_neg}")
    d.info("5 meilleures stations (MAE%) :")
    for _, r in res.head(5).iterrows():
        d.info(f"  {r['station_id']} {r['libelle'][:30]:30s} {r['mae_pct']:>5.1f}%  R²={r['r2']}")
    d.info("5 pires stations (MAE%) :")
    for _, r in res.tail(5).iterrows():
        d.info(f"  {r['station_id']} {r['libelle'][:30]:30s} {r['mae_pct']:>5.1f}%  R²={r['r2']}")

    if save:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(OUT_PATH, index=False)
        d.success(f"CSV par station : {OUT_PATH.relative_to(PROJECT_ROOT)}")

    return res


if __name__ == "__main__":
    run()
