import gc
import io

import numpy as np
import pandas as pd
from google.cloud import storage

from naviflow.ml_logic.data import get_data
from naviflow.ml_logic.feature_engineering import build_features
from naviflow.ml_logic.preprocess_xgb import prepare_xgb
from naviflow.config import GCP_PROJECT, BUCKET_NAME


def save_all_X_test(grain="station", n_clusters=4, lags=(1, 7, 30), horizon=7, test_size=0.2):

    df = get_data()
    df = build_features(df, with_cluster=(grain == "cluster"), n_clusters=n_clusters)

    group_col = "ID_LIEU" if grain == "station" else "cluster"
    storage_client = storage.Client(project=GCP_PROJECT)
    bucket = storage_client.bucket(BUCKET_NAME)

    for gid in sorted(df[group_col].dropna().unique()):
        df_group = df[df[group_col] == gid]
        if len(df_group) <= max(lags) + horizon + 1:
            continue

        X_np, _, _, dates_np = prepare_xgb(df_group, lags=lags, horizon=horizon, as_numpy=True)

        dates = np.asarray(dates_np, dtype='datetime64[ns]')
        cutoff = np.unique(dates)[int(len(np.unique(dates)) * (1 - test_size))]
        test_mask = dates >= cutoff

        df_X_test = pd.DataFrame(X_np[test_mask])
        df_X_test["JOUR"] = dates_np[test_mask]

        buffer = io.BytesIO()
        df_X_test.to_parquet(buffer, index=False)
        buffer.seek(0)
        bucket.blob(f"features_store/X_test_{gid}.parquet").upload_from_file(buffer, content_type="application/octet-stream")

        del df_group, X_np, dates_np, df_X_test
        gc.collect()


if __name__ == "__main__":
    save_all_X_test()
