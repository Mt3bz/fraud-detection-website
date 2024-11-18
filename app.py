from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pandas as pd
import logging
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup rate limiter
limiter = Limiter(key_func=get_remote_address)  # No `app` argument here
limiter.init_app(app)  # Initialize the limiter with the app here

# Load model and feature names
logging.info("Loading the model...")
logging.info("Preprocessed data for model: %s", aligned_data)

try:
    model, feature_names = joblib.load("model_with_features.pkl")
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error("Failed to load the model: %s", str(e))
    raise

# Preprocess the input data
def preprocess_data(data):
    input_data = pd.DataFrame([data])
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    return input_data

# Authentication endpoint
@app.route("/auth", methods=["POST"])
def auth():
    data = request.get_json()
    username = data.get("username")
    api_key = data.get("api_key")

    # Dummy authentication for demo
    if username == "admin" and api_key == "12345":
        return jsonify({"message": "Authentication successful"}), 200
    return jsonify({"error": "Invalid credentials"}), 401

# Fraud detection endpoint
@app.route("/predict", methods=["POST"])
@limiter.limit("5 per minute")  # Limit to 5 requests per minute per IP
def predict():
    try:
        data = request.json
        logging.info("Received data: %s", data)

        # Align input data with model features
        aligned_data = pd.DataFrame([data]).reindex(columns=feature_names, fill_value=0)
        logging.info("Aligned input data for prediction: %s", aligned_data)

        # Make predictions
        probabilities = model.predict_proba(aligned_data)[0]
        logging.info("Prediction probabilities: %s", probabilities)

        # Determine fraud status
        threshold = 0.1  # Adjust this threshold if necessary
        fraud_status = "Fraud" if probabilities[1] >= threshold else "Legitimate"

        return jsonify({
            "fraud_status": fraud_status,
            "confidence": round(probabilities[1], 2)
        })
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        return jsonify({"error": str(e)}), 500

# Error handler for 404
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

# Error handler for other HTTP exceptions
@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({"error": e.description}), e.code

# General error handler
@app.errorhandler(Exception)
def handle_exception(e):
    logging.error("Unexpected error: %s", str(e))
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
