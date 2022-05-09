import matplotlib.pyplot as plt
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization, GRU, SimpleRNN
from tensorflow.python.keras.models import Sequential
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from datetime import date
from os.path import exists
import pandas as pd
import numpy as np
from ML_Pipeline.constants import *

def build_network_lstm(embedding_layer,lstm_size):
    
    print(" Building Sequential network ")
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(lstm_size))#, return_sequences=True))
    #model.add(LSTM(100))    
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_network_GRU(embedding_layer):

    print(" Building GRU network ")
    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(100))#, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(hidden_layer_1, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_network_RNN(embedding_layer):

    print(" Building RNN network ")
    model = Sequential()
    model.add(embedding_layer)
    model.add(SimpleRNN(100))#, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(hidden_layer_1, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model