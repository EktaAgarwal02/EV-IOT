from flask import Flask,render_template, request, redirect, url_for, flash

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load # dump is used to save the model and load is used to load the model

app = Flask(__name__)



# load the saved model and preprocessing components
model = joblib.load('xgboost.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('ab.html')

@app.route("/login")
def login():
    return render_template('login.html')

if __name__ == "__main__":
    app.run(debug=True)