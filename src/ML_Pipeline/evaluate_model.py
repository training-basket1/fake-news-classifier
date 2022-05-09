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

def performance_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # plt.savefig(output_dir + image_dir+ model_type+'/' + name + "_performance.jpeg") 

def model_evaluation(model,X_test,y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    return score


def performance_report(model,testX,testy,model_name,report_dir='../output/reports/'):

    time = date.today()

    yhat_probs = model.predict(testX, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(testX, verbose=0)

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(testy, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(testy, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(testy, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(testy, yhat_classes)
    print('F1 score: %f' % f1)

    if exists(report_dir + 'report.csv'):
        total_cost_df = pd.read_csv(report_dir + 'report.csv', index_col=0)
    else:
        total_cost_df = pd.DataFrame(columns=['time', 'name', 'Precision', 'Recall', 'f1_score', 'accuracy'])
    total_cost_df = total_cost_df.append(
            {'time': time, 'name': model_name,'Precision': precision, 'Recall': recall, 'f1_score': f1,'accuracy':accuracy},
            ignore_index=True)
    total_cost_df.to_csv(report_dir + 'report.csv')
    return total_cost_df
