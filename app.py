from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pandas as pd
import logging

# Initialize Flask app
app = Flask(__name__)

# Setup rate limiter
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])

# Setup logging for business visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load XGBoost model and feature names
logging.info("Loading the XGBoost model...")
try:
    model, feature_names = joblib.load("xgboost_model_with_features.pkl")
    logging.info("XGBoost model loaded successfully!")
except Exception as e:
    logging.error("Failed to load the model: %s", str(e))
    raise

# Threshold for fraud detection (tunable based on business requirements)
FRAUD_THRESHOLD = 0.2  # Adjust sensitivity as per need


# Authentication decorator
AUTHORIZED_KEYS = {"business1": "apikey12345", "business2": "apikey67890"}

def require_api_key(f):
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("x-api-key")
        if not api_key or api_key not in AUTHORIZED_KEYS.values():
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function


# Preprocess incoming data
def preprocess_data(data):
    """
    Aligns incoming JSON data with model feature names.
    """
    input_data = pd.DataFrame([data])
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    return input_data


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint to confirm API status.
    """
    return jsonify({"status": "API is healthy"}), 200


# Authentication endpoint
@app.route("/auth", methods=["POST"])
def auth():
    """
    API authentication endpoint.
    """
    data = request.get_json()
    username = data.get("username")
    api_key = data.get("api_key")

    if AUTHORIZED_KEYS.get(username) == api_key:
        return jsonify({"message": "Authentication successful"}), 200
    return jsonify({"error": "Invalid credentials"}), 401


# Fraud detection endpoint
@app.route("/predict", methods=["POST"])
@require_api_key
@limiter.limit("10 per minute")  # Adjust rate limits for production usage
def predict():
    """
    Predicts whether a transaction is fraudulent or legitimate.
    """
    try:
        data = request.json
        logging.info("Received data: %s", data)

        # Preprocess input data
        aligned_data = preprocess_data(data)
        logging.info("Aligned input data for prediction: %s", aligned_data)

        # Model prediction
        probabilities = model.predict_proba(aligned_data)[0]
        logging.info("Prediction probabilities: %s", probabilities)

        # Fraud determination
        fraud_status = "Fraud" if probabilities[1] >= FRAUD_THRESHOLD else "Legitimate"

        return jsonify({
            "fraud_status": fraud_status,
            "confidence": round(probabilities[1], 2)
        })
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        return jsonify({"error": str(e)}), 500


# General error handler
@app.errorhandler(Exception)
def handle_exception(e):
    """
    Handle unexpected errors globally.
    """
    logging.error("Unexpected error: %s", str(e))
    return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
