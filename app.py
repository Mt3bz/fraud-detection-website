from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import joblib
import logging
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Flask-Limiter setup
limiter = Limiter(key_func=get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])

# Load the trained model
logging.info("Loading model...")
try:
    model, feature_names = joblib.load("random_forest_model_with_features2.pkl")
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error("Failed to load the model: %s", str(e))
    raise

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
    if username == "admin" and api_key == "12345":
        return jsonify({"message": "Authentication successful"}), 200
    return jsonify({"error": "Invalid credentials"}), 401

# Fraud prediction endpoint
@app.route("/predict", methods=["POST"])
@limiter.limit("5 per minute")
def predict():
    try:
        data = request.json
        logging.info("Received data: %s", data)

        # Preprocess and validate input data
        input_data = pd.DataFrame([data])
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        logging.info("Input data aligned for prediction: %s", input_data)

        # Predict using the model
        probabilities = model.predict_proba(input_data)[0]
        logging.info("Prediction probabilities: %s", probabilities)

        # Define threshold
        threshold = 0.2
        fraud_status = "Fraud" if probabilities[1] >= threshold else "Legitimate"
        return jsonify({"fraud_status": fraud_status, "confidence": round(probabilities[1], 2)})
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({"error": e.description}), e.code

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error("Unexpected error: %s", str(e))
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
