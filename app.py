from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pandas as pd
import logging
from functools import wraps
from werkzeug.exceptions import HTTPException

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask-Limiter with rate limiting
try:
    limiter = Limiter(key_func=get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])
    logging.info("Rate limiter initialized.")
except ImportError as e:
    logging.error(f"Failed to initialize limiter: {e}")
    raise

# Load the model and features
try:
    logging.info("Loading the model...")
    model, feature_names = joblib.load("random_forest_model_with_features2.pkl")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load the model: {e}")
    raise

# Authorized API keys (Replace with secure storage in production)
AUTHORIZED_KEYS = {
    "admin": "secureapikey123"
}

# Decorator for API key authentication
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("x-api-key")
        if not api_key or api_key not in AUTHORIZED_KEYS.values():
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Preprocess input data to match the model's features
def preprocess_data(data):
    input_data = pd.DataFrame([data])
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    return input_data

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is healthy"}), 200

# Authentication endpoint
@app.route("/auth", methods=["POST"])
def auth():
    data = request.json
    username = data.get("username")
    api_key = data.get("api_key")
    if AUTHORIZED_KEYS.get(username) == api_key:
        return jsonify({"message": "Authentication successful"}), 200
    return jsonify({"error": "Invalid credentials"}), 401

# Fraud detection endpoint
@app.route("/predict", methods=["POST"])
@require_api_key
@limiter.limit("5 per minute")  # Adjust rate limits as needed
def predict():
    try:
        data = request.json
        logging.info(f"Received data: {data}")

        # Preprocess data
        aligned_data = preprocess_data(data)
        logging.info(f"Aligned input data for prediction: {aligned_data}")

        # Make prediction
        probabilities = model.predict_proba(aligned_data)[0]
        fraud_status = "Fraud" if probabilities[1] >= 0.2 else "Legitimate"  # Adjust threshold as needed
        confidence = round(probabilities[1], 2)

        logging.info(f"Prediction completed. Status: {fraud_status}, Confidence: {confidence}")
        return jsonify({"fraud_status": fraud_status, "confidence": confidence})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({"error": e.description}), e.code

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unexpected error: {e}")
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
