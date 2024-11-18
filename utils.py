import pandas as pd

def preprocess_transaction_data(data, feature_names):
    """
    Preprocesses a single transaction for fraud prediction.
    
    Args:
        data (dict): JSON data of the transaction.
        feature_names (list): List of features expected by the model.
    
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction.
    """
    input_data = pd.DataFrame([data])
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    return input_data
