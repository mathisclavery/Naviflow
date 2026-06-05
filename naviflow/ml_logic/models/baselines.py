from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from keras.layers import Lambda, Input
from keras import layers
from keras.models import Model

#Import CONSTANTS
from naviflow.config import *


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

# #Baseline is cross validated with folds in the rnn_station.py
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
