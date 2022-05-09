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
import json
from ML_Pipeline.constants import *
def train_model(model,X_train,y_train,X_test, y_test):
    
    # Compile Model with loss function, 
    # optimizer and metricecs as minimum parameter
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # Train model with Train and test set data
    # Number of epochs, batch size as minimum parameter
    history = model.fit(X_train, y_train, epochs=epochs,batch_size = batch_size ,validation_split=0.2)#validation_data=(X_test, y_test))   
    return model,history


def store_model(model,file_path='../output/models/',file_name='trained_model'):
    # Store the model as json and 
    # store model weights as HDF5
    
    # serialize model to JSON
    model_json = model.to_json()
    with open(file_path+file_name+'.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save(file_path+file_name+'.h5')
    print(f"Saved model to disk in path {file_path} as {file_name + '.json'}")
    print(f"Saved weights to disk in path {file_path} as {file_name + '.h5'}")
