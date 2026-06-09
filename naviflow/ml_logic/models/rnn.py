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
import json

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
    n_stations = y_train.shape[2]  #encodes NUMBER STATIONS 100 ou 740 ...



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



def init_model_2(X_past_train, X_fut_train, y_train):


    # Branch 1 — past features
    norm_past = Normalization()
    norm_past.adapt(X_past_train) #fit the Normalisation

    # Branch 1 — processes past features with LSTM
    inp_past = Input(shape=X_past_train.shape[1:])
    x1 = norm_past(inp_past)
    x1 = layers.LSTM(64,activation='tanh',
                    return_sequences = True)(x1) #Add 'return_sequence' = True for the next layer
                    #kernel_regularizer=L1L2(l1=0.05, l2=0.05))(x1) #Regularization disabled for 1st version
    x1 = layers.LSTM(64,activation='tanh',
                    return_sequences = False)(x1)


    # Branch 2 — future covariates
    norm_fut = Normalization()
    norm_fut.adapt(X_fut_train)

    # Branch 2 — processes future covariates with LSTM
    inp_fut = Input(shape=X_fut_train.shape[1:])
    x2 = norm_fut(inp_fut)
    x2 = layers.LSTM(64,activation='tanh',
                    return_sequences = True)(x2) #Add 'return_sequence' = True for the next layer
                    #kernel_regularizer=L1L2(l1=0.05, l2=0.05))(x2) #Regularization disabled for 1st version
    x2 = layers.LSTM(64,activation='tanh',
                    return_sequences = False)(x2)

    # Merge and predict
    merged = layers.concatenate([x1, x2])
    x = layers.Dense(128, activation="relu")(merged) #Adding a layer to be less brutal
    x = layers.Dense(32, activation="relu")(x)

    #OUTPUT LAYERS SHAPE
    n_days = y_train.shape[1]   # 1 ou 7 ...
    n_stations = y_train.shape[2]  #encodes NUMBER STATIONS 100 ou 740 ...



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



def init_model_3(X_past_train, X_fut_train, y_train):


    # Branch 1 — past features
    norm_past = Normalization()
    norm_past.adapt(X_past_train) #fit the Normalisation

    # Branch 1 — processes past features with LSTM
    inp_past = Input(shape=X_past_train.shape[1:])
    x1 = norm_past(inp_past)
    x1 = layers.LSTM(64*2,activation='tanh',
                    return_sequences = True)(x1) #Add 'return_sequence' = True for the next layer
                    #kernel_regularizer=L1L2(l1=0.05, l2=0.05))(x1) #Regularization disabled for 1st version
    x1 = layers.LSTM(64*2,activation='tanh',
                    return_sequences = True)(x1)
    x1 = layers.LSTM(64*2,activation='tanh',
                    return_sequences = False)(x1)


    # Branch 2 — future covariates
    norm_fut = Normalization()
    norm_fut.adapt(X_fut_train)

    # Branch 2 — processes future covariates with LSTM
    inp_fut = Input(shape=X_fut_train.shape[1:])
    x2 = norm_fut(inp_fut)
    x2 = layers.LSTM(64*2,activation='tanh',
                    return_sequences = True)(x2) #Add 'return_sequence' = True for the next layer
                    #kernel_regularizer=L1L2(l1=0.05, l2=0.05))(x2) #Regularization disabled for 1st version
    x2 = layers.LSTM(64*2,activation='tanh',
                    return_sequences = True)(x2)
    x2 = layers.LSTM(64*2,activation='tanh',
                    return_sequences = False)(x2)

    # Merge and predict
    merged = layers.concatenate([x1, x2])
    x = layers.Dense(128, activation="relu")(merged) #Adding a layer to be less brutal
    x = layers.Dense(32, activation="relu")(x)

    #OUTPUT LAYERS SHAPE
    n_days = y_train.shape[1]   # 1 ou 7 ...
    n_stations = y_train.shape[2]  #encodes NUMBER STATIONS 100 ou 740 ...



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

    es = EarlyStopping(monitor = "val_loss",
                      patience = 100,
                      mode = "min",
                      restore_best_weights = True,
                      verbose=verbose)


    history = model.fit([X_past_train, X_fut_train], y_train,
                        validation_split = 0.2, # MISTAKE: Do not call TEST data = Dataleakage([X_past_test_station, X_fut_test_station], y_test_station),
                        shuffle = True,
                        batch_size = 32,
                        epochs = epochs,
                        callbacks = [es],
                        verbose = verbose)

    return model, history



def compute_global_score_rnn(model,X_past_test,X_fut_test, y_test, log = True):

    mean_error_list_week = []

    print('Horizon, number of days to predict', y_test.shape[1])
    print('Number of stations', y_test.shape[2])

    y_pred = model.predict([X_past_test, X_fut_test])

    for day in range(y_test.shape[1]):
        mean_error_distrib_1day = []

        #New: np.expm1 to come back to the initial #validations space
        for station_index in range(y_test.shape[2]):

            if log == True:
                mean_mae_station = np.expm1(np.abs(y_test - y_pred)[:,day,station_index]).mean()
                mean_nb_valid_station = np.expm1(np.abs(y_test)[:,day,station_index]).mean()

            else:
                mean_mae_station = np.abs(y_test - y_pred)[:,day,station_index].mean()
                mean_nb_valid_station = np.abs(y_test)[:,day,station_index].mean()


            mean_error_station = mean_mae_station/mean_nb_valid_station
            mean_error_distrib_1day.append(mean_error_station)

        #Convert to numpy + clean NaN
        mean_error_distrib_1day = np.array(mean_error_distrib_1day)
        mean_error_distrib_1day = mean_error_distrib_1day[~np.isnan(mean_error_distrib_1day)]
        mean_error_distrib_1day = mean_error_distrib_1day[np.isfinite(mean_error_distrib_1day)]
        #Compute score for 1 day
        mean_error_1day = np.mean(mean_error_distrib_1day)*100 # to get percentage
        mean_error_list_week.append(round(mean_error_1day,3))

    #Get list for 7 days
    return mean_error_list_week



def compute_stations_score_rnn(df, model, stations_dict_path, X_past_test, X_fut_test, y_test, log = True):

    y_pred = model.predict([X_past_test, X_fut_test])

    #Initiate creation of the dataframe of interest
    df_sort_validations = df[['ID_LIEU','NB_VALD_TOTAL']].groupby(['ID_LIEU']).mean()
    df_sort_validations.sort_values(by='NB_VALD_TOTAL',ascending=False,inplace=True)
    #ex: stations_dict_path= '../naviflow/dict_stations.json'
    with open(stations_dict_path, 'r') as f:
        d_str = json.load(f)
        d_dict = json.loads(d)

    d_dict_int = {int(k): v for k, v in d_dict.items()}


    #Add column to map ID_LIEU to LIBELLE_ARRET
    #TODO: Issue dictionary not with the appropriate format --> Cannot map correctly: returns NaN
    df_sort_validations['LIBELLE_ARRET'] = df_sort_validations.index.map(d_dict_int)

    #Add column of MAE per station
    mae_list_per_station = []
    day_index = 0
    for station_index, id_lieu in enumerate(df_sort_validations.index):
        if log == True:
            mae_list_per_station.append(np.expm1(np.abs(y_test - y_pred)[:,:,station_index][:,day_index]).mean())
        else:
            mae_list_per_station.append(np.abs(y_test - y_pred)[:,:,station_index][:,day_index].mean())
    df_sort_validations['MAE_PER_STATION'] = mae_list_per_station

    #Adding column relative error
    df_sort_validations['ERROR_PERCENT'] = (df_sort_validations['MAE_PER_STATION']/df_sort_validations['NB_VALD_TOTAL'])*100

    return df_sort_validations




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
