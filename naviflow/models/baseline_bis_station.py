#####################################
from keras.layers import Lambda

#Baseline is cross validated with folds in the rnn_station.py
def init_baseline_station():

    # $CHALLENGIFY_BEGIN
    model = models.Sequential()
    #In X the column with index=1 = TARGET value = 'NB_VALD_TOTAL'
    #With -1 we take the last value of the list = day before prediction
    model.add(layers.Lambda(lambda x: x[:,-1,1,None]))

    #Optimizer is still necessary in this case
    # because we build a Tensorflow 'Sequential' architecture
    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model
    # $CHALLENGIFY_END
