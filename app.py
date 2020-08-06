# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 20:26:34 2020

@author: Rohit Mehta
"""

from flask import Flask,render_template,url_for,request
import numpy as np
import pickle

loaded_model=pickle.load(open('random_forest_regression_model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    my_prediction = loaded_model.predict(final_features)
    output = my_prediction
    return render_template('home.html', prediction_text='Air Quality Index is {}'.format(output))
    


if __name__ == '__main__':
    app.run(debug=True)
    
