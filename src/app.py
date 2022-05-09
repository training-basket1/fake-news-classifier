## Importing Required Libraries
from tensorflow import keras
from flask import Flask, render_template, request
import pandas as pd
from ML_Pipeline.text_tokenizer import prepare_seqence_data
from ML_Pipeline.text_tokenizer import pad_sequence_data
from ML_Pipeline.constants import *
import pickle


#### wsgi application 
app = Flask(__name__)

#### Load Model
model = keras.models.load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news =    pd.Series(request.form['text'])
        # loading tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        message = prepare_seqence_data(news,tokenizer)
        pred =    pad_sequence_data(message,max_text_length)
        prediction = model.predict_classes(pred)
        if prediction == 0:
            output = "it looks like a real news"
        else:
            output = "it looks like a fake news"
        
        return render_template('index.html', prediction_text=output)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug = True)

