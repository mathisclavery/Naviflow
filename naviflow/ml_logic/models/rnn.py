# 20260604
# Clement Ribart
# Methods adapted from RNN challenge 'weather temperature' predictions
# Customization: Take into account Future covariates

#########################################
#IMPORTS

#Naviflow imports
from naviflow.ml_logic.data import get_data
# from naviflow.models.baseline_bis_station import init_baseline_station


#General imports
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List, Tuple, Sequence

#Imports model RNN
from keras import models, layers, Model, Input, optimizers, metrics
from keras.regularizers import L1L2
from keras.layers import Normalization
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.layers import Lambda

#Import CONSTANTS
from naviflow.config import *



##########################################
#PREPROCESSING DATA SPECIAL FOR RNN + 1 STATION
#OBJECTIVE: Prepare (X_past,X_future, y) samples to feed the model for train/val/test

###############
#Q-CR:
    # Put these methods in preprocessing.py ?
    # Do I need to call cleaning/preprocessing steps already designed by buddies?
    # quid if different Train test split depending on the model invoked?: fold_train_test_split


##############
#CR: Reminder folds are just used for cross validation
def get_folds(
    df: pd.DataFrame,
    fold_length: int,
    fold_stride: int) -> List[pd.DataFrame]:
    """
    This function slides through the Time Series dataframe of shape (n_timesteps, n_features) to create folds
    - of equal `fold_length`
    - using `fold_stride` between each fold

    Args:
        df (pd.DataFrame): Overall dataframe
        fold_length (int): How long each fold should be in rows
        fold_stride (int): How many timesteps to move forward between taking each fold

    Returns:
        List[pd.DataFrame]: A list where each fold is a dataframe within
    """

    folds = []
    for idx in range(0, len(df), fold_stride):
        # Exits the loop as soon as the last fold index would exceed the last index
        if (idx + fold_length) > len(df):
            break
        fold = df.iloc[idx:idx + fold_length, :]
        folds.append(fold)
    return folds


#Q-CR: To train/val/test on the ENTIRE dataset, can we just call fold = entire dataframe
#NB: The validation set is managed during 'fit_model' with 'validation_split'
def fold_train_test_split(fold:pd.DataFrame,
                     train_test_ratio: float,
                     input_length: int) -> Tuple[pd.DataFrame]:
    """From a fold dataframe, take a train dataframe and test dataframe based on
    the split ratio.
    - df_train should contain all the timesteps until round(train_test_ratio * len(fold))
    - df_test should contain all the timesteps needed to create all (X_test, y_test) tuples

    Args:
        fold (pd.DataFrame): A fold of timesteps
        train_test_ratio (float): The ratio between train and test 0-1
        input_length (int): How long each X_i will be

    Returns:
        Tuple[pd.DataFrame]: A tuple of two dataframes (fold_train_station, fold_test_station)
    """

    # TRAIN SET
    # ======================
    last_train_idx = round(train_test_ratio * len(fold))
    fold_train_station = fold.iloc[0:last_train_idx, :]
    fold_train_station = fold_train_station.add_prefix('_') #To make the dataframe compatible with GCP


    # TEST SET
    # ======================
    first_test_idx = last_train_idx - input_length
    fold_test_station = fold.iloc[first_test_idx:, :]
    fold_test_station = fold_test_station.add_prefix('_') #To make the dataframe compatible with GCP


    return (fold_train_station, fold_test_station)


def get_Xi_yi_station(
    fold:pd.DataFrame,
    input_length:int,
    output_length:int) -> Tuple[pd.DataFrame]:
    """given a fold, it returns one sequence (X_i, y_i) as based on the desired
    input_length and output_length with the starting point of the sequence being chosen at random based

    Args:
        fold (pd.DataFrame): A single fold
        input_length (int): How long each X_i should be
        output_length (int): How long each y_i should be

    Returns:
        Tuple[pd.DataFrame]: A tuple of two dataframes (X_i, y_i)
    """
    # $CHALLENGIFY_BEGIN
    first_possible_start = 0
    last_possible_start = len(fold) - (input_length + output_length) + 1
    random_start = np.random.randint(first_possible_start, last_possible_start)

    ############
    #FEATURES
    #PAST COVARIATES
    #NB: We keep the TARGET in X for Times Series
    X_past_i = fold.iloc[random_start:random_start+input_length]
    X_past_i.drop(columns=FUTURE_COVARIATES,inplace=True)
    #X_past_i.drop(columns=['index'],inplace=True) #TODO: Comment if fails
    X_past_i['FFM'] = X_past_i['FFM'].fillna(0) #In the orig data there are few values with Nan
    #FUTURE COVARIATES
    X_fut_i = fold.iloc[random_start:random_start+input_length+output_length]
    X_fut_i = X_fut_i[FUTURE_COVARIATES] #Keep only future covariates
    #X_fut_i.drop(columns=['index'],inplace=True) #TODO: Comment if fails

    ############
    #TARGET
    y_i = fold.iloc[random_start+input_length:
                  random_start+input_length+output_length][[TARGET]]


    return (X_past_i, X_fut_i, y_i)
    # $CHALLENGIFY_END

def get_Xi_yi(
    fold:pd.DataFrame,
    input_length:int,
    output_length:int) -> Tuple[pd.DataFrame]:

    """only difference with 'get_Xi_yi' is the management of y that has now N-stations target
    """

    # $CHALLENGIFY_BEGIN
    first_possible_start = 0
    last_possible_start = len(fold) - (input_length + output_length) + 1
    random_start = np.random.randint(first_possible_start, last_possible_start)

    ############
    #FEATURES
    #PAST COVARIATES
    #NB: We keep the TARGET in X for Times Series
    X_past_i = fold.iloc[random_start:random_start+input_length]
    X_past_i.drop(columns=FUTURE_COVARIATES,inplace=True)
    #X_past_i.drop(columns=['index'],inplace=True) #TODO: Comment if fails
    X_past_i['_FFM'] = X_past_i['_FFM'].fillna(0) #In the orig data there are few values with Nan
    #FUTURE COVARIATES
    X_fut_i = fold.iloc[random_start:random_start+input_length+output_length]
    X_fut_i = X_fut_i[FUTURE_COVARIATES] #Keep only future covariates
    #X_fut_i.drop(columns=['index'],inplace=True) #TODO: Comment if fails

    ############
    #TARGETs - Now needs to be a vector of N-stations Tickets validations
    y_i = fold.iloc[random_start+input_length:
                  random_start+input_length+output_length].drop(columns = PAST_COVARIATES)

    y_i = y_i.drop(columns=FUTURE_COVARIATES)


    return (X_past_i, X_fut_i, y_i)
    # $CHALLENGIFY_END


def get_X_y(
    fold:pd.DataFrame,
    number_of_sequences:int,
    input_length:int,
    output_length:int) -> Tuple[np.array]:
    """Given a fold generate X and y based on the number of desired sequences
    of the given input_length and output_length

    Args:
        fold (pd.DataFrame): Fold dataframe
        number_of_sequences (int): The number of X_i and y_i pairs to include
        input_length (int): Length of each X_i
        output_length (int): Length of each y_i

    Returns:
        Tuple[np.array]: A tuple of numpy arrays (X, y)
    """
    # $CHALLENGIFY_BEGIN
    X_past, X_fut, y = [], [], []

    for i in range(number_of_sequences):
        (X_past_i, X_fut_i, yi) = get_Xi_yi(fold, input_length, output_length)
        X_past.append(X_past_i)
        X_fut.append(X_fut_i)
        y.append(yi)

    return np.array(X_past), np.array(X_fut), np.array(y)


#####################################
#REAL MODEL DEFINITION + OPTIMIZATION METHODS

def init_model(X_past_train, X_fut_train, y_train):


    # Branch 1 — past features
    norm_past = Normalization()
    norm_past.adapt(X_past_train) #fit the Normalisation

    # Branch 1 — processes past features with LSTM
    inp_past = Input(shape=X_past_train.shape[1:])
    x1 = norm_past(inp_past)
    x1 = layers.LSTM(64,activation='tanh',
                    return_sequences = False)(x1)
                    #kernel_regularizer=L1L2(l1=0.05, l2=0.05))(x1) #Regularization disabled for 1st version


    # Branch 2 — future covariates
    norm_fut = Normalization()
    norm_fut.adapt(X_fut_train)

    # Branch 2 — processes future covariates with LSTM
    inp_fut = Input(shape=X_fut_train.shape[1:])
    x2 = norm_fut(inp_fut)
    x2 = layers.LSTM(32,activation='tanh',
                    return_sequences = False)(x2)
                    #kernel_regularizer=L1L2(l1=0.05, l2=0.05))(x2) #Regularization disabled for 1st version

    # Merge and predict
    merged = layers.concatenate([x1, x2])
    x = layers.Dense(128, activation="relu")(merged) #Adding a layer to be less brutal
    x = layers.Dense(32, activation="relu")(x)

    #OUTPUT LAYERS SHAPE
    #Important: A single prediction vector shape
        #Ex: If we want to have 7 days of observations in past + predict 1 day in future for 100 stations
        # (for a single observation Xi_past shape (7,107) and Xi_fut shape (8,4))
        # we need an output shape yi shape (1,100) --> y shape (N_obs,1,100)
    #Here we do not have anymore a basic scalar 'output_length' because several targets _ potentially several days
    n_days = y_train.shape[1]   # 1 ou 7 ...
    n_stations = NUMBER_STATIONS #y_train.shape[2]  # 100 ou 740 ...



    #OUTPUT LAYERS in 2 steps
    #Tim: Prefer 'linear' to 'relu' even if physically we cannot have negative NB_VALIDATION
    out = layers.Dense(n_days * n_stations, activation="linear")(x) #Actual prediction layer needs to be 1D
    out = layers.Reshape(target_shape=(n_days, n_stations),)(out) #But we want to as an output of the yi shape (1,100) --> Reshape layer :)

    #BUILD OVERALL MODEL
    model = Model(inputs=[inp_past, inp_fut], outputs=out)

       # 2 - Compiler
    # ======================
    #adam = optimizers.Adam(learning_rate=0.005)
    model.compile(loss='mse', optimizer="adam", metrics=["mae"])

    return model




def fit_model(model: tf.keras.Model,
              X_past_train,
              X_fut_train,
              y_train,
              epochs,
              verbose=1) -> Tuple[tf.keras.Model, dict]:

    # $CHALLENGIFY_BEGIN
    es = EarlyStopping(monitor = "val_loss",
                      patience = 100,
                      mode = "min",
                      restore_best_weights = True,
                      verbose=verbose)


    history = model.fit([X_past_train, X_fut_train], y_train,
                        validation_split = 0.2, # MISTAKE: Do not call TEST data = Dataleakage([X_past_test_station, X_fut_test_station], y_test_station),
                        shuffle = False,
                        batch_size = 32,
                        epochs = epochs,
                        callbacks = [es],
                        verbose = verbose)

    return model, history
    # $CHALLENGIFY_END




#####################################
#CROSS VALIDATE MODEL ON SEVERAL FOLDS - 1 STATION


# def cross_validate_baseline_and_lstm_station():
#     '''
#     This function cross-validates
#     - the "last seen value" baseline model
#     - the RNN model
#     '''

#     list_of_mae_baseline_model = []
#     list_of_mae_recurrent_model = []

#     # 0 - Creating folds
#     # =========================================
#     folds = get_folds(df, FOLD_LENGTH, FOLD_STRIDE)

#     for fold_id, fold in enumerate(folds):

#         # 1 - Train/Test split the current fold
#         # =========================================
#         (fold_train, fold_test) = fold_train_test_split(fold, TRAIN_TEST_RATIO, INPUT_LENGTH)

#         X_past_train_station, X_fut_train_station, y_train_station = get_X_y(fold_train_station, N_TRAIN, INPUT_LENGTH, OUTPUT_LENGTH)
#         X_past_test_station, X_fut_test_station, y_test_station = get_X_y(fold_test_station, N_TEST, INPUT_LENGTH, OUTPUT_LENGTH)

#         # 2 - Modelling
#         # =========================================

#         ##### Baseline Model
#         baseline_model = init_baseline_station()
#         mae_baseline = baseline_model.evaluate(X_past_test_station, y_test_station, verbose=0)[1]
#         list_of_mae_baseline_model.append(mae_baseline)
#         print("-"*50)
#         print(f"MAE baseline fold n°{fold_id} = {round(mae_baseline, 2)}")

#         ##### LSTM Model
#         model = init_model_station(X_past_train_station, X_fut_train_station, y_train_station)
#         es = EarlyStopping(monitor = "val_mae",
#                            mode = "min",
#                            patience = 2,
#                            restore_best_weights = True)
#         history = model.fit([X_past_train_station, X_fut_train_station], y_train_station,
#                             validation_split = 0.2,
#                             shuffle = False,
#                             batch_size = 32,
#                             epochs = 150, #Reduced number of epochs based on the results on 1 fold above
#                             callbacks = [es],
#                             verbose = 0)
#         res = model.evaluate([X_past_test_station, X_fut_test_station], y_test_station, verbose=0)
#         mae_lstm = res[1]
#         list_of_mae_recurrent_model.append(mae_lstm)
#         print(f"MAE LSTM fold n°{fold_id} = {round(mae_lstm, 2)}")

#         ##### Comparison LSTM vs Baseline for the current fold
#         print(f"🏋🏽‍♂️ improvement over baseline: {round((1 - (mae_lstm/mae_baseline))*100,2)} % \n")

#     return list_of_mae_baseline_model, list_of_mae_recurrent_model
