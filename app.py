from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import sklearn
import os

# Load the trained model (ensure the model is saved as a pickle file)
model_path = os.path.join(os.path.dirname(__file__), 'gbm.pkl')
model = pickle.load(open(model_path, 'rb'))  # Path to your model

app = Flask(__name__)

# Serve the HTML page on the root route
@app.route('/')
def home():
    return render_template('index.html')  # Make sure your HTML file is named index.html

# Predict route for receiving data and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json()
        if not data:
            return jsonify({"result": "No data provided"}), 400

        # Extract features from the request
        feature_names = ['gender', 'occupation', 'bmi', 'age', 'sleep_duration', 'quality_sleep', 
                         'physical_activity', 'stress_level', 'heart_rate', 'daily_steps', 
                         'systolic', 'diastolic']
        
        features = np.array([[data.get(name) for name in feature_names]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Map model prediction to corresponding label
        prediction_label = {0: "Insomnia", 1: "No Disorder", 2: "Sleep Apnea"}.get(prediction[0], "Error predicting sleep disorder")
        
        return jsonify({"result": prediction_label})

    except Exception as e:
        # Log the exception
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"result": f"Error predicting sleep disorder: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
