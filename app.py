from flask import Flask, request, jsonify
from utils import preprocess_data, validate_input, load_model

# Initialize the Flask app
app = Flask(__name__)

# Load the CatBoost model and feature names
model_path = "catboost_model_with_features.cbm"  # Update to your actual path
feature_names = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

print("Loading the model...")
model = load_model(model_path)
print("Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for fraud detection."""
    try:
        # Parse input JSON
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Validate input data
        errors = validate_input(data, feature_names)
        if errors:
            return jsonify({"error": "Invalid input", "details": errors}), 400

        # Preprocess the input data
        input_data = preprocess_data(data, feature_names)

        # Make predictions
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]

        # Format response
        result = {
            "prediction": "Fraud" if prediction[0] == 1 else "Legitimate",
            "confidence": float(prediction_proba[0])
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": "An error occurred during prediction", "details": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "Healthy"}), 200
