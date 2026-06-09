
from naviflow.ml_logic.data import get_data


from naviflow.ml_logic.preprocess_rnn import create_rnn_dataframe
from naviflow.ml_logic.preprocess_rnn import get_folds, fold_train_test_split
from naviflow.ml_logic.preprocess_rnn import get_Xi_yi, get_X_y

from naviflow.ml_logic.models.rnn import init_model_3, fit_model, compute_global_score_rnn, compute_stations_score_rnn


#Import CONSTANTS
from naviflow.config import *

def train(df, epochs = 10):


    #preprocess to build dataframe of interest
    df_stations = create_rnn_dataframe(df,log=True)

    #Split data
    (fold_train, fold_test) = fold_train_test_split(df_stations, TRAIN_TEST_RATIO, INPUT_LENGTH)

    #Generate samples
    X_past_train, X_fut_train, y_train = get_X_y(fold_train, N_TRAIN, INPUT_LENGTH, OUTPUT_LENGTH)

    #Init model
    model = init_model_3(X_past_train, X_fut_train, y_train)

    #fit model
    model, history = fit_model(
    model,
    X_past_train,
    X_fut_train,
    y_train,
    epochs=epochs
    )

    return model, history


#def evaluate():
    # X_past_test, X_fut_test, y_test = get_X_y(fold_test, N_TEST, INPUT_LENGTH, OUTPUT_LENGTH)

#def pred():


if __name__ == '__main__':
    preprocess()
    train()
    #evaluate()
    #pred()
