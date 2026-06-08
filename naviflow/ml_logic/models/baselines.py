from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from keras.layers import Lambda, Input
from keras import layers
from keras.models import Model

#Import CONSTANTS
from naviflow.config import *


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from naviflow.config import TARGET


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from naviflow.config import TARGET


def run_baseline_weekday(df_group, horizon=7, lags=(1, 7, 30),
                         test_size=0.2, group="ID_LIEU", date="JOUR"):
    """Baseline 'meme jour de la semaine derniere'.

    Pour predire J+h, on prend la valeur observee a J+(h-7).
    Renvoie un dict avec, en plus des metriques brutes, une cle 'flat' :
    un dict plat {mae_j1, pct_j1, mae_j2, pct_j2, ...} pret pour le CSV.
    pct_jh = MAE de l'horizon h / moyenne reelle des validations * 100.
    """
    df = df_group.copy()
    df[date] = pd.to_datetime(df[date])
    df = df.sort_values([group, date])

    horizons = list(range(1, horizon + 1))

    for h in horizons:
        df[f"target_j{h}"] = df.groupby(group)[TARGET].shift(-h)
    for h in horizons:
        df[f"pred_j{h}"] = df.groupby(group)[TARGET].shift(7 - h)

    target_cols = [f"target_j{h}" for h in horizons]
    pred_cols   = [f"pred_j{h}"   for h in horizons]
    df = df.dropna(subset=target_cols + pred_cols)

    dates = df[date].to_numpy(dtype="datetime64[ns]")
    unique_days = np.unique(dates)
    cutoff = unique_days[int(len(unique_days) * (1 - test_size))]
    test_mask = dates >= cutoff

    Y_test    = df.loc[test_mask, target_cols].to_numpy()
    Pred_test = df.loc[test_mask, pred_cols].to_numpy()

    mae_raw = mean_absolute_error(Y_test, Pred_test, multioutput='raw_values')

    # Moyenne reelle par horizon (pour le % d'erreur)
    true_mean = Y_test.mean(axis=0)

    # Dict plat pour le CSV : une colonne MAE + une colonne % par horizon
    flat = {}
    for h, mae_h, mean_h in zip(horizons, mae_raw, true_mean):
        flat[f"mae_j{h}"] = round(float(mae_h))
        flat[f"pct_j{h}"] = round(float(mae_h / mean_h * 100), 1) if mean_h else None
    flat["n_test"] = len(Y_test)

    return {
        'mae': float(mae_raw.mean()),
        'mae_per_h': {h: float(m) for h, m in zip(horizons, mae_raw)},
        'flat': flat,
        'y_pred': Pred_test,
        'y_test': Y_test,
    }







# #####################################

# prediction of the day = value day-7
#ex: Predict value of Monday = value of previous monday - naive dumb prediction taking into account week periodicity
def init_baseline_rnn(X_past_train):

    # Branch 1 — processes past features with LSTM
    inp_past = Input(shape=X_past_train.shape[1:])

    out = layers.Lambda(lambda x: x[:,-7:-6,:NUMBER_STATIONS])(inp_past)

    #BUILD OVERALL MODEL
    model = Model(inputs=inp_past, outputs=out)

       # 2 - Compiler
    # ======================
    #adam = optimizers.Adam(learning_rate=0.005)
    model.compile(loss='mse', optimizer="adam", metrics=["mae"])


    return model
