import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras
import string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # Loop the inputs and create a pair of an array with an output
    for i in range(len(series) - window_size):
        X.append(series[i:(i + window_size)])
        y.append(series[i + window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    # Layer 1 = LSTM 5 hidden units with shape of windows_size, 1
    model.add(LSTM(5, input_shape=(window_size,1)))
    # Layer 2 = Fully connected
    model.add(Dense(1, activation='tanh'))

    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    # Added single space to valid punctuation
    punctuation = ['!', ',', '.', ':', ';', '?', ' ']


    # Created a list of ascii
    valid = string.ascii_lowercase

    # Loop the text and join if valid or if it's in punctuation
    text = "".join(i for i in text if i in valid or i in punctuation )

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    # Loop the inputs and create a pair of an array with an output
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:(i + window_size)])
        outputs.append(text[i + window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    # Layer 1
    model.add(LSTM(200, input_shape = (window_size, num_chars))) 

    # Layer 2
    model.add(Dense(num_chars))

    # Layer 3
    model.add(Activation('softmax'))

    return model
