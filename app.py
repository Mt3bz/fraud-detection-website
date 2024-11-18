from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pandas as pd
import logging
from utils import preprocess_data, validate_input

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure rate limiting
limiter = Limiter(get_remote_address)
limiter.init_app(app)

# Load model
logging.info("Loading the model...")
try:
    model, feature_names = joblib.load("random_forest_model_with_features2.pkl")
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Failed to load the model: {str(e)}")
    raise

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "API is live!"}), 200

# Fraud detection endpoint
@app.route("/predict", methods=["POST"])
@limiter.limit("5 per minute")
def predict():
    try:
        data = request.json
        logging.info(f"Received data: {data}")
        
        # Validate input
        errors = validate_input(data, feature_names)
        if errors:
            return jsonify({"error": "Invalid input", "details": errors}), 400
        
        # Preprocess input
        aligned_data = preprocess_data(data, feature_names)
        logging.info(f"Aligned input data for prediction: {aligned_data}")

        # Make prediction
        probabilities = model.predict_proba(aligned_data)[0]
        logging.info(f"Prediction probabilities: {probabilities}")
        
        # Fraud threshold
        threshold = 0.2
        fraud_status = "Fraud" if probabilities[1] >= threshold else "Legitimate"

        return jsonify({
            "fraud_status": fraud_status,
            "confidence": round(probabilities[1], 2)
        })
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Error handler
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
