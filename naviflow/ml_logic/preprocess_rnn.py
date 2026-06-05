"""Preprocessing SPECIFIQUE au RNN (sequences temporelles 3D).

Part du DataFrame enrichi commun (feature_engineering.build_features) et produit
des tenseurs (X_past, X_fut, y) pour alimenter le LSTM a deux branches.

Architecture en deux niveaux, pour anticiper la refonte mono -> multi-stations :
  - BRIQUE  : prepare les sequences d'UNE serie temporelle continue
              (filtre station, tri, folds, sequences past/future). Reutilisable.
  - ORCHESTRATION : selectionne la/les station(s). Aujourd'hui mono-station ;
              la version multi iterera sur les stations en reutilisant la brique.

Scaling : fait ICI via sklearn (RobustScaler), FIT SUR LE TRAIN DU FOLD
uniquement, puis applique au test du meme fold. Le RNN n'a donc plus besoin
de couche Normalization interne.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from naviflow.config import (
    PAST_COVARIATES,
    FUTURE_COVARIATES,
    TARGET,
    FOLD_LENGTH,
    FOLD_STRIDE,
    TRAIN_TEST_RATIO,
    INPUT_LENGTH,
    OUTPUT_LENGTH,
    N_TRAIN,
    N_TEST,
)

# Colonnes continues a scaler (les binaires IS_*/is_* restent en 0/1)
RNN_CONTINUOUS_COLS = ["RR", "TN", "TX", "TM", "FFM", "amplitude", "log_RR",
                       "mois_sin", "mois_cos", TARGET]


# --------------------------------------------------------------------------- #
# Selection d'une serie mono-station, triee
# --------------------------------------------------------------------------- #
def get_station_series(df, id_lieu):
    """Extrait la serie temporelle d'UNE station, triee par JOUR, index reset.

    Garde uniquement les colonnes utiles au RNN : TARGET + past + future
    covariates (dedupliquees, dans un ordre stable).
    """
    cols = [TARGET] + list(dict.fromkeys(PAST_COVARIATES + FUTURE_COVARIATES))
    serie = (df[df["ID_LIEU"] == id_lieu]
             .sort_values("JOUR")
             .reset_index(drop=True))
    return serie[cols]


# --------------------------------------------------------------------------- #
# Folds (cross-validation temporelle)
# --------------------------------------------------------------------------- #
def get_folds(df: pd.DataFrame, fold_length: int, fold_stride: int) -> List[pd.DataFrame]:
    """Decoupe la serie en folds glissants de longueur fold_length, pas fold_stride."""
    folds = []
    for idx in range(0, len(df), fold_stride):
        if (idx + fold_length) > len(df):
            break
        folds.append(df.iloc[idx:idx + fold_length, :])
    return folds


def fold_train_test_split(fold: pd.DataFrame,
                          train_test_ratio: float,
                          input_length: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Coupe un fold en train / test chronologiquement.

    Le test redemarre `input_length` lignes avant la coupure pour pouvoir
    reconstruire des sequences completes a cheval sur la frontiere.
    """
    last_train_idx = round(train_test_ratio * len(fold))
    fold_train = fold.iloc[0:last_train_idx, :]
    fold_test  = fold.iloc[last_train_idx - input_length:, :]
    return fold_train, fold_test


# --------------------------------------------------------------------------- #
# Scaling fold-wise (fit sur le train du fold)
# --------------------------------------------------------------------------- #
def scale_fold(fold_train, fold_test):
    """Scale les colonnes continues : fit sur le train du fold, transform les deux.

    Evite le leakage temporel : le scaler ne voit jamais le test pendant le fit.
    Les colonnes binaires (covariates calendaires) ne sont pas touchees.
    """
    fold_train = fold_train.copy()
    fold_test  = fold_test.copy()

    cols = [c for c in RNN_CONTINUOUS_COLS if c in fold_train.columns]
    scaler = RobustScaler().fit(fold_train[cols])
    fold_train[cols] = scaler.transform(fold_train[cols])
    fold_test[cols]  = scaler.transform(fold_test[cols])

    return fold_train, fold_test, scaler


# --------------------------------------------------------------------------- #
# Sequences (X_past, X_fut, y)
# --------------------------------------------------------------------------- #
def get_Xi_yi(fold: pd.DataFrame, input_length: int, output_length: int):
    """Tire UNE sequence (X_past_i, X_fut_i, y_i) a un point de depart aleatoire.

    - X_past : input_length jours, covariates passees + TARGET (memoire).
    - X_fut  : input_length + output_length jours, covariates futures connues
               (calendrier), sans la TARGET (c'est ce qu'on predit).
    - y      : output_length jours de TARGET a predire.
    """
    last_possible_start = len(fold) - (input_length + output_length) + 1
    random_start = np.random.randint(0, last_possible_start)

    # Past covariates (on garde TARGET dans le passe)
    X_past_i = fold.iloc[random_start:random_start + input_length].copy()
    X_past_i = X_past_i.drop(columns=FUTURE_COVARIATES)

    # Future covariates (calendaires connus a l'avance), sans la TARGET
    X_fut_i = fold.iloc[random_start:random_start + input_length + output_length].copy()
    X_fut_i = X_fut_i.drop(columns=PAST_COVARIATES + [TARGET])

    # Target
    y_i = fold.iloc[random_start + input_length:
                    random_start + input_length + output_length][[TARGET]]

    return X_past_i, X_fut_i, y_i


def get_X_y(fold: pd.DataFrame, number_of_sequences: int,
            input_length: int, output_length: int):
    """Genere number_of_sequences triplets (X_past, X_fut, y) -> arrays numpy 3D."""
    X_past, X_fut, y = [], [], []
    for _ in range(number_of_sequences):
        xp, xf, yi = get_Xi_yi(fold, input_length, output_length)
        X_past.append(xp)
        X_fut.append(xf)
        y.append(yi)
    return np.array(X_past), np.array(X_fut), np.array(y)


# --------------------------------------------------------------------------- #
# Orchestration : preparation complete pour UNE station
# --------------------------------------------------------------------------- #
def prepare_rnn_station(df, id_lieu,
                        fold_length=FOLD_LENGTH, fold_stride=FOLD_STRIDE,
                        train_test_ratio=TRAIN_TEST_RATIO,
                        input_length=INPUT_LENGTH, output_length=OUTPUT_LENGTH,
                        n_train=N_TRAIN, n_test=N_TEST):
    """Prepare les sequences train/test d'UNE station, fold par fold.

    Renvoie une liste de dicts (un par fold) :
      {X_past_train, X_fut_train, y_train, X_past_test, X_fut_test, y_test, scaler}

    C'est la brique reutilisable : la future version multi-stations iterera
    sur les stations en appelant cette fonction pour chacune.
    """
    serie = get_station_series(df, id_lieu)
    folds = get_folds(serie, fold_length, fold_stride)

    results = []
    for fold in folds:
        fold_train, fold_test = fold_train_test_split(fold, train_test_ratio, input_length)
        fold_train, fold_test, scaler = scale_fold(fold_train, fold_test)

        Xp_tr, Xf_tr, y_tr = get_X_y(fold_train, n_train, input_length, output_length)
        Xp_te, Xf_te, y_te = get_X_y(fold_test,  n_test,  input_length, output_length)

        results.append({
            "X_past_train": Xp_tr, "X_fut_train": Xf_tr, "y_train": y_tr,
            "X_past_test":  Xp_te, "X_fut_test":  Xf_te, "y_test":  y_te,
            "scaler": scaler,
        })
    return results
