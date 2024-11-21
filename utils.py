import pandas as pd
import joblib
from catboost import CatBoostClassifier

def preprocess_data(data, feature_names):
    """Preprocess input data to match the model's feature structure."""
    input_data = pd.DataFrame([data])
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    return input_data

def validate_input(data, feature_names):
    """Validate the input JSON against expected features."""
    errors = []
    for feature in feature_names:
        if feature not in data:
            errors.append(f"Missing feature: {feature}")
    return errors

def load_model(model_path):
    """Load the trained CatBoost model."""
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
