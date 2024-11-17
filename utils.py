import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(features, feature_names):
    """
    Preprocess input features for prediction.
    Ensures all required features are present and aligned.
    """
    # Convert input JSON to DataFrame
    input_data = pd.DataFrame([features])

    # Add missing columns with default value 0
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Default value for missing features

    # Drop extra columns not expected by the model
    input_data = input_data[feature_names]

    # Log the aligned input data for debugging
    print("Aligned input data for prediction:", input_data)

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(input_data)
    return scaled_features
