from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pandas as pd
import logging
from werkzeug.exceptions import HTTPException

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup rate limiter
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])

# Load model and feature names
logging.info("Loading the Random Forest model...")
try:
    model, feature_names = joblib.load("random_forest_model_with_features.pkl")
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

    # Dummy authentication for demonstration
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

        aligned_data = preprocess_data(data)
        logging.info("Aligned input data for prediction: %s", aligned_data)

        probabilities = model.predict_proba(aligned_data)[0]
        logging.info("Prediction probabilities: %s", probabilities)

        threshold = 0.5  # Fraud detection threshold
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
