from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import sklearn
import os

# Load the trained model (ensure the model is saved as a pickle file)
model = pickle.load(open('gbm.pkl', 'rb'))  # Path to your model

app = Flask(__name__)

# Serve the HTML page on the root route
@app.route('/')
def home():
    return render_template('index.html')  # Make sure your HTML file is named index.html

# Predict route for receiving data and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.get_json()

    # Extract features from the request
    gender = data['gender']
    occupation = data['occupation']
    bmi = data['bmi']
    age = data['age']
    sleep_duration = data['sleep_duration']
    quality_sleep = data['quality_sleep']
    physical_activity = data['physical_activity']
    stress_level = data['stress_level']
    heart_rate = data['heart_rate']
    daily_steps = data['daily_steps']
    systolic = data['systolic']
    diastolic = data['diastolic']

    # Prepare the data for prediction (ensure it's in the right format)
    features = np.array([[gender, occupation, bmi, age, sleep_duration, quality_sleep,
                          physical_activity, stress_level, heart_rate, daily_steps, systolic, diastolic]])

    # Make prediction
    prediction = model.predict(features)

    # Map model prediction to corresponding label
    prediction_label = {0: "Insomnia", 1: "No Disorder", 2: "Sleep Apnea"}.get(prediction[0], "Error predicting sleep disorder")

    return jsonify({"result": prediction_label})

if __name__ == "__main__":
    app.run(debug=True)
