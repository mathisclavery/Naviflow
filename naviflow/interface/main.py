"""Pipeline d'entrainement XGBoost de bout en bout.

Enchaine : get_data -> build_features -> boucle sur les groupes -> prepare_xgb
(en numpy) -> entrainement -> sauvegarde modele + metriques.

Le GRAIN d'entrainement est parametrable :
  - grain='station' : un modele par station (743 modeles).
  - grain='cluster' : un modele par cluster (4 modeles), chaque modele voyant
                      toutes les stations de son cluster.

Memoire : le gros DataFrame de toutes les stations n'est JAMAIS converti en un
seul numpy array. On le decoupe par groupe (chaque groupe fait quelques milliers
de lignes), on convertit ce petit morceau en numpy juste avant fit, et on libere
apres chaque iteration.
"""

import gc

from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.preprocess_xgb import prepare_xgb
from naviflow.ml_logic.models.sklearn_models import run_xgboost
from naviflow import registry


def train_all(grain="station", n_clusters=4, lags=(1, 7, 30), horizon=None,
              n_iter=50, save=True, verbose=True):
    """Entraine un XGBoost par groupe (station ou cluster) et sauvegarde tout.

    Parametres
    ----------
    grain : 'station' (boucle sur ID_LIEU) ou 'cluster' (boucle sur cluster).
    n_clusters : nombre de clusters si grain='cluster' (alimente build_features).
    lags : decalages temporels passes a prepare_xgb.
    horizon : si renseigne, entraine a predire J+horizon (sinon jour courant).
    n_iter : iterations du RandomizedSearch dans run_xgboost.
    save : si True, sauvegarde chaque modele + le CSV de metriques.
    verbose : log la progression.

    Renvoie
    -------
    dict {group_id: metrics} (metriques par groupe, sans les modeles).
    """
    # 1. Donnees + features (UNE seule fois, sur tout le dataset)
    with_cluster = (grain == "cluster")
    df = get_data()
    df = build_features(df, with_cluster=with_cluster, n_clusters=n_clusters)

    # 2. Colonne qui definit les groupes
    group_col = "ID_LIEU" if grain == "station" else "cluster"
    group_ids = sorted(g for g in df[group_col].dropna().unique())

    if verbose:
        print(f"Grain={grain} | {len(group_ids)} groupes a entrainer\n")

    results = {}

    # 3. Boucle sur les groupes
    for i, gid in enumerate(group_ids, 1):
        df_group = df[df[group_col] == gid]

        # Garde-fou : un groupe trop court ne peut pas produire de lags exploitables
        if len(df_group) <= max(lags) + 1:
            if verbose:
                print(f"[{i}/{len(group_ids)}] {grain} {gid} ignore (trop peu de donnees)")
            continue

        # Conversion numpy au dernier moment, sur un petit morceau
        X_np, y_np, feature_names = prepare_xgb(
            df_group, lags=lags, horizon=horizon, as_numpy=True
        )

        res = run_xgboost(X_np, y_np, n_iter=n_iter)

        # On ne garde que les metriques (pas y_pred/y_test, lourds) dans le dict
        results[gid] = {"mae": res["mae"], "r2": res["r2"], "mae_cv": res["mae_cv"],
                        "n_samples": len(y_np)}

        if save:
            registry.save_model(res["model"], gid, grain=grain)

        if verbose:
            print(f"[{i}/{len(group_ids)}] {grain} {gid} — MAE={res['mae']:.0f} "
                  f"R2={res['r2']:.3f} (n={len(y_np)})")

        # Liberation memoire avant l'iteration suivante
        del df_group, X_np, y_np, res
        gc.collect()

    # 4. Recap des metriques dans un CSV unique
    if save and results:
        registry.save_results(results, grain=grain)
        if verbose:
            print(f"\nMetriques sauvegardees : {registry.RESULTS_CSV}")

    return results


if __name__ == "__main__":
    # Par defaut : un modele par station, prediction du jour courant
    train_all(grain="station")
