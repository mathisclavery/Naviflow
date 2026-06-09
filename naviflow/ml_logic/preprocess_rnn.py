
##########################################
#CR: PREPROCESSING DATA SPECIAL FOR RNN
#OBJECTIVE: Prepare (X_past,X_future, y) samples to feed the model for train/val/test

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


#####################
#NAVIFLOW IMPORTS

from naviflow.ml_logic.sources.meteo import load_meteo
from naviflow.ml_logic.sources.calendrier import load_calendrier


#Import CONSTANTS
from naviflow.config import *

##################
#OTHER GENERAL Imports

import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List, Tuple, Sequence



# Colonnes continues a scaler (les binaires IS_*/is_* restent en 0/1)
RNN_CONTINUOUS_COLS = ["RR", "TN", "TX", "TM", "FFM", "amplitude", "log_RR",
                       "mois_sin", "mois_cos", TARGET]



def create_rnn_dataframe(df, log = True):
    df = df.copy()
    #Tableau croisé dynamique
    df = df.pivot(index='JOUR', columns='ID_LIEU', values='NB_VALD_TOTAL')
    df.fillna(0,inplace=True)

    #Data log conversion
    if log == True:
        df = np.log1p(df)
    else:
        pass

    #Build final data frame
    df_meteo = load_meteo()
    df_calendrier = load_calendrier()

    df_stations = df.merge(df_meteo, on="JOUR", how="left").merge(df_calendrier, on="JOUR", how="left")
    df_stations.fillna(0,inplace=True)

    ###########
    #Manager time
    df_stations['JOUR'] = pd.to_datetime(df_stations['JOUR'])

    #df_stations['day_of_week'] = df_stations['JOUR'].dt.dayofweek
    #Commented because affluence discontinuity at this level
    #Days go from 1 to 7 in cycles
    # df_stations['day_sin'] = np.sin(2 * np.pi * df_stations['day'] / 7)
    # df_stations['day_cos'] = np.cos(2 * np.pi * df_stations['day'] / 7)

    df_stations['month'] = df_stations['JOUR'].dt.month
    #Months values go from 1 to 12 with cyclicity
    df_stations['month_sin'] = np.sin(2 * np.pi * df_stations['month'] / 12)
    df_stations['month_cos'] = np.cos(2 * np.pi * df_stations['month'] / 12)
    df_stations.drop(columns=['month'],inplace=True)

    #Year has NO cyclicity in feature engineering: Years only increase iteratively
    #However obviously years have physical cyclicity: Every year we have summer, winter, fall, spring
    #df_stations['year'] = df_stations['JOUR'].dt.year

    df_stations.drop(columns='JOUR',inplace=True)

    ###########
    #Manage Day_offs
    df_stations['IS_WEEKEND'] = df_stations['IS_WEEKEND'].replace({True: 1, False: 0})
    df_stations['IS_FERIE'] = df_stations['IS_FERIE'].replace({True: 1, False: 0})
    df_stations['IS_VACANCES'] = df_stations['IS_VACANCES'].replace({True: 1, False: 0})
    df_stations['IS_PONT'] = df_stations['IS_PONT'].replace({True: 1, False: 0})


    return df_stations


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
    #print('fold.shape',fold.shape)

    first_possible_start = 0
    last_possible_start = len(fold) - (input_length + output_length) + 1
    random_start = np.random.randint(first_possible_start, last_possible_start)

    ############
    #FEATURES
    #PAST COVARIATES
    #NB: We keep the TARGET in X for Times Series
    X_past_i = fold.iloc[random_start:random_start+input_length]
    #print('X_past_i.shape',X_past_i.shape)
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
                  random_start+input_length+output_length]

    #print('y_i.shape',y_i.shape)

    y_i = y_i.drop(columns = PAST_COVARIATES)
    #print('y_i.shape',y_i.shape)

    y_i = y_i.drop(columns=FUTURE_COVARIATES)
    #print('y_i.shape',y_i.shape)


    return (X_past_i, X_fut_i, y_i)


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
    X_past, X_fut, y = [], [], []

    for i in range(number_of_sequences):
        (X_past_i, X_fut_i, yi) = get_Xi_yi(fold, input_length, output_length)
        X_past.append(X_past_i)
        X_fut.append(X_fut_i)
        y.append(yi)

    return np.array(X_past), np.array(X_fut), np.array(y)
