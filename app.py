from math import e
from flask import Flask, render_template, request, redirect
import pandas as pd
import os
from prophet import Prophet
from datetime import datetime
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib.pyplot as plt
import io 
import base64

app = Flask(__name__)

import mysql.connector

print("Starting connection...")  # Debugging line
import mysql.connector

mydb = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='',
    database='bitcoin'  # No database
)

print("Connection successful!")

mycur = mydb.cursor(dictionary=True)  # Keep dictionary mode

# Load dataset and prepare the Prophet model
file_path = "BTC-USD.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Close']]
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
print(df.tail())
import pandas as pd

# Assume df is your original dataset
new_data = pd.DataFrame({'ds': ['2025-02-25'], 'y': [(91418 + 88751) / 2]})  # Simple average
df = pd.concat([df, new_data], ignore_index=True)

df['ds'] = pd.to_datetime(df['ds'])  # Ensure correct datetime format
df = df.sort_values('ds')  # Keep the order correct

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

            sql = 'INSERT INTO users (name, email, password, Address) VALUES (%s, %s, %s, %s)'
            mycur.execute(sql, (name, email, hashed_password, address))
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
                return redirect('/home')
            return render_template('login.html', msg='Incorrect password!')

        return render_template('login.html', msg='User does not exist. Please register.')

    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/viewdata')
def viewdata():
    df = pd.read_csv('BTC-USD.csv').tail(100)
    table_html = df.to_html(classes='table table-striped table-hover', index=False)
    return render_template('viewdata.html', table=table_html)

def generate_plot(forecast_data):
    """ Generates a Matplotlib plot and returns it as a base64 string. """
    df = pd.DataFrame(forecast_data)

    plt.figure(figsize=(8, 4))
    plt.plot(df["ds"], df["yhat"], marker='o', linestyle='-', label="Predicted Price", color="blue")
    plt.fill_between(df["ds"], df["yhat_lower"], df["yhat_upper"], color="gray", alpha=0.2, label="Confidence Interval")

    plt.xlabel("Date")
    plt.ylabel("Bitcoin Price (USD)")
    plt.title("Bitcoin Price Prediction")
    plt.legend()
    plt.xticks(rotation=45)

    # Convert the plot to a BytesIO buffer
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', bbox_inches="tight")
    img_io.seek(0)
    plt.close()

    # Convert to base64
    encoded_img = base64.b64encode(img_io.getvalue()).decode("utf-8")
    return encoded_img


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    forecast = None
    if request.method == "POST":
        start_date_str = request.form['start_date']
        end_date_str = request.form['end_date']

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        if end_date <= start_date:
            return render_template('prediction.html', forecast=[], msg="End date must be after start date.")

        future = pd.date_range(start=start_date, end=end_date, freq='D')
        future_df = pd.DataFrame(future, columns=['ds'])

        forecast = model.predict(future_df)
           

    return render_template('prediction.html',forecast=forecast.to_dict(orient='records') if forecast is not None else [])

if __name__ == '__main__':
    app.run(debug=True)
 