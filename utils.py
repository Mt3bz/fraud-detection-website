import pandas as pd
import logging

def preprocess_data(data, feature_names):
    """
    Preprocess input data to match the model's feature structure.
    - Ensures that all required features are present.
    - Fills missing features with 0.
    """
    try:
        input_data = pd.DataFrame([data])
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        return input_data
    except Exception as e:
        logging.error(f"Error during data preprocessing: {str(e)}")
        raise ValueError("Data preprocessing failed")

def validate_input(data, feature_names):
    """
    Validate the input JSON against expected features.
    - Checks for missing features.
    - Ensures correct data types.
    """
    errors = []
    for feature in feature_names:
        if feature not in data:
            errors.append(f"Missing feature: {feature}")

    for feature, value in data.items():
        if feature in feature_names:
            if isinstance(value, (int, float)) or value is None:
                continue
            elif feature == "type" and not isinstance(value, int):
                errors.append(f"Feature '{feature}' must be an integer")
            else:
                errors.append(f"Invalid data type for feature '{feature}': {type(value).__name__}")

    if errors:
        logging.warning(f"Input validation errors: {errors}")
    return errors
