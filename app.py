from math import e
from flask import Flask, render_template, request, redirect, session, flash
import pandas as pd
import os
from prophet import Prophet
from datetime import datetime
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib.pyplot as plt
import io 
import base64
from werkzeug.utils import secure_filename 
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import Input
from keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(4)  # Use 4 parallel threads
tf.config.threading.set_intra_op_parallelism_threads(4)


app = Flask(__name__)
app.secret_key = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
DEFAULT_DP = 'default_dp.png'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

import mysql.connector

print("Starting connection...")  # Debugging line

mydb = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='',
    database='bitcoin'  # No database
)

print("Connection successful!")

mycur = mydb.cursor(dictionary=True)  # Keep dictionary mode
 

# # Load dataset and prepare the Prophet model
# Load dataset
file_path = "BTC-USD.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Close']]
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
df = df.sort_values('ds')

# Prophet Model
model = Prophet(changepoint_prior_scale=0.5)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.fit(df)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        address = request.form['Address']

        if password == confirmpassword:
            sql = 'SELECT * FROM users WHERE email = %s'
            mycur.execute(sql, (email,))
            data = mycur.fetchone()

            if data:
                return render_template('registration.html', msg='User already registered!')

            # Hash password before storing
            hashed_password = generate_password_hash(password)

            sql = 'INSERT INTO users (name, email, password, Address, dp) VALUES (%s, %s, %s, %s, %s)'
            mycur.execute(sql, (name, email, hashed_password, address, DEFAULT_DP))
            mydb.commit()
            return render_template('registration.html', msg='User registered successfully!')

        return render_template('registration.html', msg='Passwords do not match!')
    
    return render_template('registration.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        sql = 'SELECT * FROM users WHERE email=%s'
        mycur.execute(sql, (email,))
        data = mycur.fetchone()

        if data:
            stored_password = data['password']
            if check_password_hash(stored_password, password):
                session['email'] = email
                return redirect('/home')
            return render_template('login.html', msg='Incorrect password!')

        return render_template('login.html', msg='User does not exist. Please register.')

    return render_template('login.html')

@app.route('/profile')
def profile():
    if 'email' not in session:
        flash("Please log in first")
        return redirect('/login')
    sql = 'SELECT * FROM users WHERE email=%s'
    mycur.execute(sql, (session['email'],))
    user = mycur.fetchone()
    return render_template('profile.html', user=user)

@app.route('/edit', methods=['POST'])
def edit():
    if 'email' not in session:
        flash("Please log in first")
        return redirect('/login')

    name = request.form['name']
    address = request.form['address']

    sql = 'UPDATE users SET name=%s, Address=%s WHERE email=%s'
    mycur.execute(sql, (name, address, session['email']))
    mydb.commit()

    if 'dp' in request.files:
        file = request.files['dp']
        if file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            sql = 'UPDATE users SET dp=%s WHERE email=%s'
            mycur.execute(sql, (filename, session['email']))
            mydb.commit()

    flash("Profile updated successfully")
    return redirect('/profile')

@app.route('/logout')
def logout():
    session.pop('email', None)
    flash("Logged out successfully")
    return redirect('/login')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/viewdata')
def viewdata():
    df = pd.read_csv('BTC-USD.csv').tail(100)
    table_html = df.to_html(classes='table table-striped table-hover', index=False)
    return render_template('viewdata.html', table=table_html)
 









"""MODEL ALGORITHMS"""
# LSTM Model Loading
def load_lstm_model():
    try:
        model = load_model('lstm_model.h5')
        scaler = np.load('scaler.npy', allow_pickle=True).item()
        return model, scaler
    
    except e:
        print(e)
        print("No model found, retraining LSTM...")
        return create_rnn_lstm_model(df['y'])

# LSTM Model Creation
def create_rnn_lstm_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(30, len(data_scaled)):
        X.append(data_scaled[i-30:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=64, verbose=1)
    model.save('lstm_model.h5')
    np.save('scaler.npy', scaler)
    return model, scaler

# ARIMA Model Creation
def create_arima_model(data):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit
@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'GET':
        return render_template('prediction.html')
    
    start_date = datetime.strptime(request.form['start_date'], "%Y-%m-%d")
    end_date = datetime.strptime(request.form['end_date'], "%Y-%m-%d")
    future = pd.date_range(start=start_date, end=end_date, freq='D')
    final_predictions = np.zeros(len(future))
    total_weight = 0

    weights = {
        'Prophet': 1,  # Reduced weight for Prophet
        'LSTM': 2,     # Reduced weight for LSTM
        'ARIMA': 5     # Increased weight for ARIMA
    }

    model_predictions = {}

    # # Prophet Prediction
    future_df = pd.DataFrame({'ds': future})
    prophet_forecast = model.predict(future_df)
    model_predictions['Prophet'] = prophet_forecast['yhat'].values
    final_predictions += weights['Prophet'] * prophet_forecast['yhat'].values
    total_weight += weights['Prophet']

    # LSTM Prediction
    lstm_model, scaler = load_lstm_model()
    last_30_days = scaler.transform(df['y'].iloc[-30:].values.reshape(-1, 1))
    last_30_days = last_30_days.reshape(1, 30, 1)
    predictions = []
    for _ in range(len(future)):
        pred = lstm_model.predict(last_30_days)
        predictions.append(pred[0][0])
        last_30_days = np.append(last_30_days[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions).flatten()
    model_predictions['LSTM'] = predictions
    final_predictions += weights['LSTM'] * predictions
    total_weight += weights['LSTM']

    # ARIMA Prediction
    arima_model = create_arima_model(df['y'])
    arima_prediction = arima_model.forecast(steps=len(future)).values
    model_predictions['ARIMA'] = arima_prediction
    final_predictions += weights['ARIMA'] * arima_prediction
    total_weight += weights['ARIMA']

    final_predictions /= total_weight
    forecast = pd.DataFrame({'ds': future, 'yhat': final_predictions})

    return render_template('prediction.html', forecast=forecast.to_dict(orient='records'), model_predictions=model_predictions)



def create_accuracy_graph(lstm_pred, arima_fitted, prophet_forecast, lstm_accuracy, arima_accuracy, prophet_accuracy):
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'], df['y'], label='Actual', color='blue')
    plt.plot(df['ds'][30:len(lstm_pred) + 30], lstm_pred, label=f'LSTM ({lstm_accuracy:.2f}%)', color='green')
    plt.plot(df['ds'][1:len(arima_fitted) + 1], arima_fitted, label=f'ARIMA ({arima_accuracy:.2f}%)', color='red')
    plt.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label=f'Prophet ({prophet_accuracy:.2f}%)', color='orange')
    plt.legend()
    plt.title('Model Predictions and Accuracy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return plot_url












@app.route('/accuracy')
def accuracy():
    # Prophet model evaluation
    prophet_forecast = model.predict(df[['ds']])
    prophet_mae = mean_absolute_error(df['y'], prophet_forecast['yhat'])
    prophet_mse = mean_squared_error(df['y'], prophet_forecast['yhat'])
    prophet_accuracy = 100 - (prophet_mae / df['y'].mean() * 100)

    # LSTM model evaluation
    lstm_model, scaler = load_lstm_model()
    data_scaled = scaler.transform(df['y'].values.reshape(-1, 1))
    X, y = [], []
    for i in range(30, len(data_scaled)):
        X.append(data_scaled[i-30:i, 0])
        y.append(data_scaled[i, 0])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    lstm_pred = lstm_model.predict(X)
    lstm_pred = scaler.inverse_transform(lstm_pred)
    actual_values_lstm = df['y'][30:len(lstm_pred) + 30].values
    lstm_mae = mean_absolute_error(actual_values_lstm, lstm_pred.flatten())
    lstm_mse = mean_squared_error(actual_values_lstm, lstm_pred.flatten())
    lstm_accuracy = 100 - (lstm_mae / df['y'].mean() * 100)

    # ARIMA model evaluation
    arima_model = create_arima_model(df['y'])
    arima_fitted = arima_model.fittedvalues[:len(df['y'][1:])]
    actual_values_arima = df['y'][1:len(arima_fitted) + 1]
    arima_mae = mean_absolute_error(actual_values_arima, arima_fitted)
    arima_mse = mean_squared_error(actual_values_arima, arima_fitted)
    arima_accuracy = 100 - (arima_mae / df['y'].mean() * 100)

    # Create visualization
    plot_url = create_accuracy_graph(lstm_pred, arima_fitted, prophet_forecast, lstm_accuracy, arima_accuracy, prophet_accuracy)
    # Create dictionary with all metrics to pass to template
    metrics = {
        'prophet': {
            'accuracy': round(prophet_accuracy, 2),
            'mae': round(prophet_mae, 4),
            'mse': round(prophet_mse, 4)
        },
        'lstm': {
            'accuracy': round(lstm_accuracy, 2),
            'mae': round(lstm_mae, 4),
            'mse': round(lstm_mse, 4)
        },
        'arima': {
            'accuracy': round(arima_accuracy, 2),
            'mae': round(arima_mae, 4),
            'mse': round(arima_mse, 4)
        }
    }

    return render_template(
        'accuracy.html', 
        plot_url=plot_url, 
        metrics=metrics
    )
if __name__ == '__main__':
    app.run(debug=True)
