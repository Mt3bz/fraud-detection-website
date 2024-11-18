from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pandas as pd
import logging
from utils import preprocess_data, validate_input
from werkzeug.exceptions import HTTPException

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up rate limiter
limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)

# Load the model and feature names
logging.info("Loading the model...")
try:
    model, feature_names = joblib.load("random_forest_model_with_features2.pkl")
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Failed to load the model: {e}")
    raise

# Preprocess and validate input data
@app.route("/predict", methods=["POST"])
@limiter.limit("5 per minute")  # Limit to 5 requests per minute per IP
def predict():
    try:
        # Parse JSON input
        data = request.json
        logging.info(f"Received data: {data}")

        # Validate input data
        errors = validate_input(data, feature_names)
        if errors:
            return jsonify({"error": "Invalid input", "details": errors}), 400

        # Preprocess input data
        aligned_data = preprocess_data(data, feature_names)
        logging.info(f"Aligned input data for prediction: {aligned_data}")

        # Make prediction
        probabilities = model.predict_proba(aligned_data)[0]
        logging.info(f"Prediction probabilities: {probabilities}")

        # Apply threshold
        threshold = 0.2  # Customize threshold as needed
        fraud_status = "Fraud" if probabilities[1] >= threshold else "Legitimate"

        return jsonify({
            "fraud_status": fraud_status,
            "confidence": round(probabilities[1], 2)
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is healthy"}), 200

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
    logging.error(f"Unexpected error: {e}")
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
