from flask import Flask, request, render_template
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd

# Load your trained XGBoost model (assuming it was saved as a .pkl file)
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize the Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # Display the input form

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Extract the form data
    fever = request.form.get('Fever', type=int)
    cough = request.form.get('Cough', type=int)
    fatigue = request.form.get('Fatigue', type=int)
    db = request.form.get('DB', type=int)
    age = request.form.get('Age', type=int)
    gender = request.form.get('Gender', type=int)
    bp = request.form.get('BP', type=int)
    cl = request.form.get('CL', type=int)

    # Create a list of feature values from form input
    features = [fever, cough, fatigue, db, age, gender, bp, cl]

    # Define the expected feature names (must match training data)
    feature_names = ['Fever', 'Cough', 'Fatigue', 'DB', 'Age', 'Gender', 'BP', 'CL']

    # Convert the input to a DataFrame with the correct feature names
    input_data = pd.DataFrame([features], columns=feature_names)

    # Convert the DataFrame to DMatrix (required by XGBoost for prediction)
    dmatrix_input = xgb.DMatrix(input_data)

    # Make prediction (using raw scores if you want probabilities)
    y_pred_raw = model.predict(dmatrix_input)

    # Since this is binary classification, convert the raw scores to binary predictions
    y_pred = (y_pred_raw > 0.5).astype(int)

    # Display the result as Positive or Negative
    result = "Positive" if y_pred[0] == 1 else "Negative"

    # Render the result on the result.html template
    return render_template('result.html', result=result)


if __name__ == "__main__":
    app.run(debug=True, port=5002)
