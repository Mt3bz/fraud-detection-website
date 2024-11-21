import json
from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
from utils import preprocess_data, validate_input

# Load the model and feature names
print("Loading the CatBoost model...")
model, features = load("catboost_model_with_features.cbm")
print("Model and features loaded successfully!")

# Define threshold for fraud detection
THRESHOLD = 0.2  # Confidence threshold for fraud classification

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON data from request
        input_data = request.json
        if not input_data:
            return jsonify({"error": "Invalid input. No data provided."}), 400

        # Validate input
        errors = validate_input(input_data, features)
        if errors:
            return jsonify({"error": "Invalid input.", "details": errors}), 400

        # Preprocess input
        input_df = preprocess_data(input_data, features)
        print(f"Input Data for Prediction: {input_df}")

        # Predict probabilities
        probabilities = model.predict_proba(input_df)
        confidence = probabilities[0][1]  # Probability of being fraudulent

        # Generate prediction
        prediction = "Fraudulent" if confidence >= THRESHOLD else "Legitimate"

        # Format response
        response = {
            "confidence": f"{confidence * 100:.2f}%",  # Convert confidence to percentage
            "prediction": prediction,
            "input_details": input_data,  # Include original input for business transparency
        }
        print(f"Prediction Response: {response}")

        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "API is up and running!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
