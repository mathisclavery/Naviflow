"""Point d'entrée CLI du projet Naviflow.

Usage :
    python -m naviflow.interface.main download
    python -m naviflow.interface.main preprocess
    python -m naviflow.interface.main baseline
    python -m naviflow.interface.main train
    python -m naviflow.interface.main evaluate
    python -m naviflow.interface.main pred
    python -m naviflow.interface.main upload_raw

Ou via Makefile :
    make download / make preprocess / make baseline / make train / ...
"""

import sys
from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.preprocess_xgb import prepare_xgb
from naviflow.ml_logic.models.sklearn_models import run_xgboost
from naviflow.ml_logic.models.baselines import run_baseline_mean, run_baseline_lag
from naviflow import registry


# ------------------------------------------------------------------ #
# 0. DOWNLOAD — télécharge raw_data/ depuis GCS
# ------------------------------------------------------------------ #
def download():
    """Télécharge les données brutes depuis GCS vers raw_data/ local."""
    from naviflow.gcp.gcs_loader import download_raw_data
    download_raw_data()


# ------------------------------------------------------------------ #
# 1. PREPROCESS
# ------------------------------------------------------------------ #
def preprocess():
    """Charge les données brutes, construit les features et affiche un aperçu."""
    print("→ Chargement des données...")
    df = get_data()
    print(f"  {len(df):,} lignes chargées ({df['ID_LIEU'].nunique()} stations)")

    print("→ Feature engineering...")
    df = build_features(df)
    print(f"  Colonnes disponibles : {list(df.columns)}")
    print("✓ Preprocessing terminé.")
    return df


# ------------------------------------------------------------------ #
# 2. BASELINE
# ------------------------------------------------------------------ #
def baseline(station_id=None):
    """Entraîne et évalue les modèles naïfs (moyenne + lag J-7) sur une station.

    Paramètres
    ----------
    station_id : ID_LIEU de la station à tester.
                 Si None, utilise la station avec le plus de données.
    """
    print("→ Chargement des données...")
    df = get_data()
    df = build_features(df)

    # Sélection de la station
    if station_id is None:
        station_id = df.groupby("ID_LIEU").size().idxmax()
        print(f"  Aucune station spécifiée — utilisation de ID_LIEU={station_id}")

    df_station = df[df["ID_LIEU"] == station_id]
    print(f"  {len(df_station)} jours disponibles pour la station {station_id}")

    # Préparation des features (avec lags pour la baseline lag_7)
    X, y = prepare_xgb(df_station, lags=(1, 7, 30), horizon=1)

    print("\n→ Baseline 1 : prédiction par la moyenne")
    run_baseline_mean(X, y)

    print("\n→ Baseline 2 : persistance J-7 (même jour semaine dernière)")
    run_baseline_lag(X, y, lag_col="lag_7")

    print("\n✓ Baseline terminée — ces scores sont le plancher à battre.")


# ------------------------------------------------------------------ #
# 3. TRAIN
# ------------------------------------------------------------------ #
def train(grain="station", lags=(1, 7, 30), horizon=1, n_iter=10):
    """Entraîne un XGBoost par groupe et sauvegarde les résultats."""
    print("→ Chargement et feature engineering...")
    df = get_data()
    with_cluster = (grain == "cluster")
    df = build_features(df, with_cluster=with_cluster)

    groups    = df["ID_LIEU"].unique() if grain == "station" else df["cluster"].unique()
    group_col = "ID_LIEU" if grain == "station" else "cluster"

    all_results = {}

    for i, group_id in enumerate(groups):
        df_group = df[df[group_col] == group_id]
        X, y = prepare_xgb(df_group, lags=lags, horizon=horizon)

        if len(X) < 50:
            print(f"  [SKIP] {group_col}={group_id} — pas assez de données ({len(X)} lignes)")
            continue

        print(f"→ [{i+1}/{len(groups)}] Entraînement {grain} {group_id} ({len(X)} lignes)...")
        result = run_xgboost(X, y, n_iter=n_iter)

        registry.save_model(result["model"], group_id=group_id, grain=grain)

        all_results[group_id] = {
            "mae":         result["mae"],
            "r2":          result["r2"],
            "mae_cv":      result["mae_cv"],
            "best_params": str(result["best_params"]),
        }

    registry.save_results(all_results, grain=grain)
    print(f"\n✓ Entraînement terminé — {len(all_results)} modèles sauvegardés.")
    return all_results


# ------------------------------------------------------------------ #
# 4. EVALUATE
# ------------------------------------------------------------------ #
def evaluate():
    """Affiche un résumé des métriques sauvegardées dans results_xgb.csv."""
    import pandas as pd
    from naviflow.registry import RESULTS_CSV

    if not RESULTS_CSV.exists():
        print("✗ Aucun résultat trouvé. Lance d'abord : make train")
        return

    df = pd.read_csv(RESULTS_CSV)
    print(f"\n{'='*50}")
    print(f"RÉSULTATS — {len(df)} groupes")
    print(f"{'='*50}")
    print(f"MAE moyenne  : {df['mae'].mean():.0f}")
    print(f"MAE médiane  : {df['mae'].median():.0f}")
    print(f"R² moyen     : {df['r2'].mean():.3f}")
    print(f"R² médian    : {df['r2'].median():.3f}")
    print(f"\nTop 5 meilleures stations (MAE) :")
    print(df.nsmallest(5, "mae")[["group_id", "mae", "r2"]].to_string(index=False))
    print(f"\nTop 5 moins bonnes stations (MAE) :")
    print(df.nlargest(5, "mae")[["group_id", "mae", "r2"]].to_string(index=False))
    return df


# ------------------------------------------------------------------ #
# 5. PRED
# ------------------------------------------------------------------ #
def pred(group_id=None, grain="station", lags=(1, 7, 30), horizon=1):
    """Charge un modèle sauvegardé et prédit sur les dernières données."""
    print("→ Chargement des données...")
    df = get_data()
    df = build_features(df, with_cluster=(grain == "cluster"))

    group_col = "ID_LIEU" if grain == "station" else "cluster"

    if group_id is None:
        group_id = df[group_col].iloc[0]
        print(f"  Aucun group_id fourni — utilisation de {group_col}={group_id}")

    df_group = df[df[group_col] == group_id]
    X, y     = prepare_xgb(df_group, lags=lags, horizon=horizon)
    model    = registry.load_model(group_id=group_id, grain=grain)
    y_pred   = model.predict(X)

    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y, y_pred)
    r2  = r2_score(y, y_pred)

    print(f"\n{'='*50}")
    print(f"PRÉDICTION — {grain} {group_id}")
    print(f"{'='*50}")
    print(f"MAE  : {mae:.0f} validations")
    print(f"R²   : {r2:.3f}")
    print(f"Dernière prédiction : {y_pred[-1]:.0f} validations (J+{horizon})")
    return y_pred


# ------------------------------------------------------------------ #
# 6. UPLOAD RAW
# ------------------------------------------------------------------ #
def upload_raw():
    """Upload les données brutes locales vers GCS."""
    import naviflow.gcp.upload_raw_data as uploader
    print("→ Upload des données brutes vers GCS...")
    uploader.upload_folder(uploader.LOCAL_FOLDER, uploader.BUCKET_FOLDER)
    print("✓ Upload terminé.")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
COMMANDS = {
    "download":   download,
    "preprocess": preprocess,
    "baseline":   baseline,
    "train":      train,
    "evaluate":   evaluate,
    "pred":       pred,
    "upload_raw": upload_raw,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage : python -m naviflow.interface.main <commande>")
        print(f"Commandes disponibles : {', '.join(COMMANDS)}")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()

