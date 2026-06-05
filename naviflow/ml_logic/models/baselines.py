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


def run_baseline_weekday(df_group, horizon=7, lags=(1, 7, 30),
                         test_size=0.2, group="ID_LIEU", date="JOUR"):
    """Baseline 'meme jour de la semaine derniere'.

    Pour predire J+h, on prend la valeur observee a J+(h-7) : le meme jour
    de la semaine, 7 jours avant le jour cible. Ex. J+5 -> J-2, J+1 -> J-6,
    J+7 -> J-7.

    df_group : DataFrame d'UN groupe (station ou cluster), pas encore passé
               par prepare_xgb (on a besoin de la serie complete + JOUR).
    Renvoie le meme format de dict que run_xgboost (mae, r2, mae_per_h...).
    """
    df = df_group.copy()
    df[date] = pd.to_datetime(df[date])
    df = df.sort_values([group, date])

    horizons = list(range(1, horizon + 1))

    # Cibles reelles : target_jh = valeur a J+h (shift negatif)
    for h in horizons:
        df[f"target_j{h}"] = df.groupby(group)[TARGET].shift(-h)

    # Predictions baseline : pred_jh = valeur a J+(h-7) (shift de 7-h)
    # shift(+k) ramene une valeur passee ; ici k = 7 - h
    for h in horizons:
        df[f"pred_j{h}"] = df.groupby(group)[TARGET].shift(7 - h)

    target_cols = [f"target_j{h}" for h in horizons]
    pred_cols   = [f"pred_j{h}"   for h in horizons]

    # On droppe les lignes ou une cible OU une prediction manque
    # (debut de serie pour les preds, fin de serie pour les targets)
    df = df.dropna(subset=target_cols + pred_cols)

    # Split temporel par date (meme logique que run_xgboost)
    dates = df[date].to_numpy(dtype="datetime64[ns]")
    unique_days = np.unique(dates)
    cutoff = unique_days[int(len(unique_days) * (1 - test_size))]
    test_mask = dates >= cutoff

    Y_test    = df.loc[test_mask, target_cols].to_numpy()
    Pred_test = df.loc[test_mask, pred_cols].to_numpy()

    mae = mean_absolute_error(Y_test, Pred_test)
    r2  = r2_score(Y_test, Pred_test)

    mae_raw = mean_absolute_error(Y_test, Pred_test, multioutput='raw_values')
    r2_raw  = r2_score(Y_test, Pred_test, multioutput='raw_values')
    mae_per_h = {h: float(m) for h, m in zip(horizons, mae_raw)}
    r2_per_h  = {h: float(r) for h, r in zip(horizons, r2_raw)}

    return {
        'mae': mae,
        'r2': r2,
        'mae_per_h': mae_per_h,
        'r2_per_h': r2_per_h,
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
