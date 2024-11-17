from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and feature names
print("Loading the model...")
model, feature_names = joblib.load("model_with_features.pkl")
print("Model loaded successfully!")

# Preprocess the data
def preprocess_data(data, feature_names):
    # Ensure the input has all required features
    input_data = pd.DataFrame([data])
    input_data = input_data.reindex(columns=feature_names, fill_value=0)  # Fill missing features with 0
    return input_data

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received data:", data)
        
        # Align input data with model features
        aligned_data = pd.DataFrame([data], columns=feature_names)
        print("Aligned input data for prediction:", aligned_data)
        
        # Make prediction
        probabilities = model.predict_proba(aligned_data)[0]
        print("Prediction probabilities:", probabilities)
        
        # Adjust threshold for fraud detection
        threshold = 0.1  # Lowered for higher sensitivity
        fraud_status = "Fraud" if probabilities[1] >= threshold else "Legitimate"
        
        # Log feature contributions
        feature_contributions = model.feature_importances_
        print("Feature Contributions:", dict(zip(feature_names, feature_contributions)))
        
        return jsonify({"fraud_status": fraud_status, "confidence": round(probabilities[1], 2)})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
