from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from keras.layers import Lambda, Input
from keras import layers
from keras.models import Model

#Import CONSTANTS
from naviflow.config import *

import numpy as np


def run_baseline_mean(X, y, test_size=0.2, random_state=67):
    """
    Baseline naive : prédit toujours la moyenne de y_train.
    Sert de point de comparaison minimum pour les autres modèles.

    Args:
        X: features preprocessées (DataFrame issu de preprocess_to_dataframe)
        y: target (Series NB_VALD_TOTAL)
        test_size: proportion du test set (défaut 0.2)
        random_state: seed pour la reproductibilité

    Returns:
        dict avec mae, r2, y_pred, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    y_pred = [y_train.mean()] * len(y_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("=" * 50)
    print("BASELINE — Prédiction de la moyenne")
    print("=" * 50)
    print(f"MAE test        : {mae:.0f}")
    print(f"R²              : {r2:.3f}")
    print(f"Erreur relative : {mae / y_test.mean() * 100:.1f}%")

    return {
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred,
        'y_test': y_test,
    }

def run_baseline_lag(X, y, test_size=0.2, random_state=67, lag_col='lag_7'):
    """
    Baseline de persistance : prédit l'affluence d'un lag passé.

    Par défaut utilise lag_7 (même jour la semaine dernière) — c'est la
    meilleure baseline naïve pour des données journalières avec saisonnalité
    hebdomadaire, car elle capture automatiquement le pattern lundi/dimanche.

    C'est une baseline plus exigeante que la simple moyenne : elle force le
    modèle ML à prouver qu'il apporte mieux que "demain ressemble à la
    semaine dernière".

    Args:
        X: features preprocessées contenant la colonne de lag
        y: target (Series NB_VALD_TOTAL)
        test_size: proportion du test set (défaut 0.2)
        random_state: seed pour la reproductibilité
        lag_col: colonne de lag à utiliser comme prédiction (défaut 'lag_7')

    Returns:
        dict avec mae, r2, y_pred, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Prédiction = valeur du lag (ex: affluence du même jour semaine dernière)
    y_pred = X_test[lag_col]

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("=" * 50)
    print(f"BASELINE — Persistance ({lag_col})")
    print("=" * 50)
    print(f"MAE test        : {mae:.0f}")
    print(f"R2              : {r2:.3f}")
    print(f"Erreur relative : {mae / y_test.mean() * 100:.1f}%")

    return {
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred,
        'y_test': y_test,
    }







# #####################################
# BASELINE RNN

#STRATEGY 1
#prediction of 1 day = value day-7
    #ex: Predict value of Monday = value of previous monday
    # naive dumb prediction taking into account week periodicity

#STRATEGY 2
#prediction of several days of the week (2 to 7 max: M,T,W,T,F,S,S)

def init_baseline_rnn(X_past, X_fut, y): #Advice Tim: ask 'X_fut' also as parameter even if not used here, to keep same structure

    # Branch 1 — Just input of the shape of the Xi
    inp_past = Input(shape=X_past.shape[1:])
    print('X_past.shape[1:]',X_past.shape[1:])

    if y.shape[1]-7 < 0: #ie if #days to predict < 7 days
        out = layers.Lambda(lambda x: x[:,-7:y.shape[1]-7,:NUMBER_STATIONS])(inp_past)
    #Manage exception where we want to predict exactly 7 days (never more in practice)
    else:
        out = layers.Lambda(lambda x: x[:,-7:,:NUMBER_STATIONS])(inp_past)

    print('out.shape =', out.shape)

    #BUILD OVERALL MODEL
    # Generic definition, but the Baseline is only a Sequential model
    model = Model(inputs=inp_past, outputs=out)

       # 2 - Compiler
    # ======================
    #adam = optimizers.Adam(learning_rate=0.005)
    model.compile(loss='mse', optimizer="adam", metrics=["mae"])


    return model


def compute_score_baseline_rnn(model,X_past, y):

    y_pred = model.predict(X_past)

    mean_error_list_baseline_week = []

    print('y.shape[1]',y.shape[1])

    for day in range(y.shape[1]):
        mean_error_distrib_baseline_1day = []

        for station_index in range(NUMBER_STATIONS):
            mean_mae_station = np.abs(y - y_pred)[:,day,station_index].mean()
            mean_nb_valid_station = np.abs(y)[:,day,station_index].mean()
            mean_error_station = mean_mae_station/mean_nb_valid_station
            mean_error_distrib_baseline_1day.append(mean_error_station)
        mean_error_baseline_1day = np.mean(mean_error_distrib_baseline_1day)
        mean_error_list_baseline_week.append(round(mean_error_baseline_1day,2))

    return mean_error_list_baseline_week
