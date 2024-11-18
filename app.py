from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pandas as pd
import logging

# Initialize Flask application
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup Limiter with correct initialization
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
limiter.init_app(app)  # Attach the limiter to the app

# Load the model and features
logging.info("Loading the model...")
try:
    model, feature_names = joblib.load("random_forest_model_with_features.pkl")
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Failed to load the model: {e}")
    raise

# Preprocess the input data
def preprocess_data(data):
    """Preprocess incoming JSON data to align with model features."""
    input_data = pd.DataFrame([data])
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    return input_data

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    """Check the health of the application."""
    return jsonify({"status": "API is healthy"}), 200

# Authentication endpoint
@app.route("/auth", methods=["POST"])
def auth():
    """Dummy authentication endpoint for demonstration."""
    data = request.get_json()
    username = data.get("username")
    api_key = data.get("api_key")

    # Replace with a proper authentication mechanism in production
    if username == "admin" and api_key == "12345":
        return jsonify({"message": "Authentication successful"}), 200
    return jsonify({"error": "Invalid credentials"}), 401

# Fraud prediction endpoint
@app.route("/predict", methods=["POST"])
@limiter.limit("5 per minute")
def predict():
    """Predict fraud or legitimate transactions."""
    try:
        data = request.json
        logging.info(f"Received data: {data}")
        
        # Preprocess data
        aligned_data = preprocess_data(data)
        logging.info(f"Aligned input data for prediction: {aligned_data}")

        # Perform prediction
        probabilities = model.predict_proba(aligned_data)[0]
        logging.info(f"Prediction probabilities: {probabilities}")

        # Fraud detection logic
        threshold = 0.1  # Fraud detection threshold
        fraud_status = "Fraud" if probabilities[1] >= threshold else "Legitimate"

        return jsonify({
            "fraud_status": fraud_status,
            "confidence": round(probabilities[1], 2)
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Error handler for 404
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

# General error handler
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle unexpected errors."""
    logging.error(f"Unexpected error: {e}")
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
