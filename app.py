from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pandas as pd
import logging
import os
from werkzeug.exceptions import HTTPException
from utils import preprocess_data, validate_input

app = Flask(__name__)

# Load environment variables for sensitive data
API_KEYS = os.getenv("API_KEYS", "admin:12345").split(",")  # Example: "user1:apikey1,user2:apikey2"
AUTHORIZED_KEYS = {k.split(":")[0]: k.split(":")[1] for k in API_KEYS}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

# Setup rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
limiter.init_app(app)  # Attach the limiter to the app

# Load model and features
logging.info("Loading the model...")
try:
    model, feature_names = joblib.load("randomforest_model_with_features.pkl")
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Failed to load the model: {str(e)}")
    raise

# Authentication endpoint
@app.route("/auth", methods=["POST"])
def auth():
    data = request.get_json()
    username = data.get("username")
    api_key = data.get("api_key")

    if username in AUTHORIZED_KEYS and AUTHORIZED_KEYS[username] == api_key:
        return jsonify({"message": "Authentication successful"}), 200
    logging.warning("Unauthorized access attempt")
    return jsonify({"error": "Invalid credentials"}), 401

# Fraud detection endpoint
@app.route("/predict", methods=["POST"])
@limiter.limit("5 per minute")  # Limit to 5 requests per minute per IP
def predict():
    try:
        data = request.json
        logging.info(f"Received data: {data}")

        # Validate input
        validation_errors = validate_input(data, feature_names)
        if validation_errors:
            logging.error(f"Input validation failed: {validation_errors}")
            return jsonify({"error": "Invalid input", "details": validation_errors}), 400

        # Preprocess and predict
        aligned_data = preprocess_data(data, feature_names)
        probabilities = model.predict_proba(aligned_data)[0]
        fraud_status = "Fraud" if probabilities[1] >= 0.2 else "Legitimate"  # Adjusted threshold to 0.2

        logging.info(f"Prediction completed. Status: {fraud_status}, Confidence: {probabilities[1]:.2f}")
        return jsonify({"fraud_status": fraud_status, "confidence": round(probabilities[1], 2)})
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
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
    logging.error(f"Unexpected error: {str(e)}")
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
