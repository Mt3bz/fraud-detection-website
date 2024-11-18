from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import logging
from werkzeug.exceptions import HTTPException
from utils import predict_fraud

app = Flask(__name__)

# Rate limiter setup
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API key authorization
AUTHORIZED_KEYS = {
    "user1": "apikey12345",
    "user2": "apikey67890"
}

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("x-api-key")
        if not api_key or api_key not in AUTHORIZED_KEYS.values():
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

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

# Prediction endpoint
@app.route("/predict", methods=["POST"])
@require_api_key
@limiter.limit("5 per minute")
def predict():
    try:
        data = request.json
        logging.info("Received data: %s", data)
        result = predict_fraud(data)
        logging.info("Prediction result: %s", result)
        return jsonify(result), 200
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        return jsonify({"error": str(e)}), 500

# Error handling
@app.errorhandler(404)
def not_found(error):
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
