from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("fetal_model.pkl")
scaler = joblib.load("fetal_scaler.pkl")  # Load the saved scaler

# Feature names expected by the model
feature_names = [
    "baseline_value", "accelerations", "fetal_movement", "uterine_contractions",
    "light_decelerations", "severe_decelerations", "prolonged_decelerations",
    "abnormal_short_term_variability", "mean_short_term_variability",
    "percentage_abnormal_long_term_variability", "mean_long_term_variability",
    "histogram_width", "histogram_min", "histogram_max", "histogram_peaks",
    "histogram_zeroes", "histogram_mode", "histogram_mean", "histogram_median",
    "histogram_variance", "histogram_tendency"
]

# Add a debug route to test the pathological case directly
@app.route("/test_pathological")
def test_pathological():
    # This is the pathological test case
    data = [134, 0.001, 0, 0.013, 0.008, 0, 0.003, 29, 6.3, 0, 0, 150, 50, 200, 6, 3, 71, 107, 106, 215, 0]
    
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([data], columns=feature_names)
    
    # Scale the data using the saved scaler
    scaled_data = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    
    # Get prediction probabilities
    proba = model.predict_proba(scaled_data)[0]
    
    # Map prediction to class label
    labels = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    result = labels.get(prediction, "Unknown")
    
    return jsonify({
        "input_data": data,
        "prediction": int(prediction),
        "result": result,
        "probabilities": {
            "Normal (1)": float(proba[0]),
            "Suspect (2)": float(proba[1]) if len(proba) > 1 else 0,
            "Pathological (3)": float(proba[2]) if len(proba) > 2 else 0
        }
    })

@app.route("/")
def home():
    return render_template("index.html", feature_names=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        data = [float(request.form[feature]) for feature in feature_names]
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data], columns=feature_names)
        
        # Scale the data using the saved scaler
        scaled_data = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        
        # Get prediction probabilities
        proba = model.predict_proba(scaled_data)[0]
        
        # Map prediction to class label
        labels = {1: "Normal", 2: "Suspect", 3: "Pathological"}
        result = labels.get(prediction, "Unknown")
        
        # Format probabilities for display
        probabilities = {
            "Normal": f"{proba[0]*100:.1f}%",
            "Suspect": f"{proba[1]*100:.1f}%" if len(proba) > 1 else "0%",
            "Pathological": f"{proba[2]*100:.1f}%" if len(proba) > 2 else "0%"
        }
        
        return jsonify({
            "result": result,
            "probabilities": probabilities
        })
    except ValueError as e:
        return jsonify({"error": "Invalid input! Please ensure all values are numeric.", "details": str(e)})
    except Exception as e:
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)})

if __name__ == "__main__":
    app.run(debug=True)