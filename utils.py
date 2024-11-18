import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the trained model and preprocessor
model_pipeline = joblib.load("fraud_detection_pipeline.pkl")

def preprocess_data(data):
    """
    Preprocess the incoming data to align with the model's expected input.
    """
    # Ensure input data matches the required format
    input_data = pd.DataFrame([data])
    return input_data

def predict_fraud(data):
    """
    Predict whether a transaction is fraudulent using the trained model pipeline.
    """
    try:
        # Preprocess data
        aligned_data = preprocess_data(data)
        
        # Predict fraud status and probabilities
        probabilities = model_pipeline.predict_proba(aligned_data)[0]
        fraud_status = "Fraud" if probabilities[1] >= 0.1 else "Legitimate"  # Adjust the threshold if needed
        return {"fraud_status": fraud_status, "confidence": round(probabilities[1], 2)}
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")
