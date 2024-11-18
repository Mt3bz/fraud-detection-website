from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import joblib
import pandas as pd
import logging
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# Setup rate limiter
limiter = Limiter(get_remote_address, default_limits=["200 per day", "50 per hour"])
limiter.init_app(app)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and feature names
logging.info("Loading the model...")
try:
    model, feature_names = joblib.load("model_with_features.pkl")
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error("Failed to load the model: %s", str(e))
    raise

# Authorized API keys (store in a secure environment or database in production)
AUTHORIZED_KEYS = {
    "user1": "apikey12345",
    "user2": "apikey67890"
}

# API key decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("x-api-key")
        if not api_key or api_key not in AUTHORIZED_KEYS.values():
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Preprocess the input data
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
    data = request.get_json()
    username = data.get("username")
    api_key = data.get("api_key")

    if AUTHORIZED_KEYS.get(username) == api_key:
        return jsonify({"message": "Authentication successful"}), 200
    return jsonify({"error": "Invalid credentials"}), 401

# Fraud detection endpoint
@app.route("/predict", methods=["POST"])
@require_api_key
@limiter.limit("5 per minute")  # Limit to 5 requests per minute per IP
def predict():
    try:
        data = request.json
        logging.info("Received data: %s", data)

        aligned_data = preprocess_data(data)
        logging.info("Aligned input data for prediction: %s", aligned_data)

        probabilities = model.predict_proba(aligned_data)[0]
        logging.info("Prediction probabilities: %s", probabilities)

        threshold = 0.1  # Fraud detection threshold
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
