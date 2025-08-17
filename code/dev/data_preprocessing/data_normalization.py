from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalize_data(data, price_value="Close"):
    """
    Normalize the data
    Args:
        data: The data to normalize (can be DataFrame or Series)
        price_value: The price value to normalize (only used if data is DataFrame)
    Returns:
        scaled_data: The normalized data
        scaler: The scaler used to normalize the data
    """
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Handle both DataFrame and Series inputs
    if hasattr(data, 'columns'):  # DataFrame
        values = data[price_value].values.reshape(-1, 1)
    else:  # Series
        values = data.values.reshape(-1, 1)
    
    scaled_data = scaler.fit_transform(values)
    return scaled_data, scaler

def denormalize_data(scaled_data, scaler):
    """
    Denormalize the data
    """
    return scaler.inverse_transform(scaled_data.reshape(-1, 1))

# gonna apply z-score normalization and rolling window scaling
# to the data
# z-score normalization:
#   - handle data leakage
#   - robust to outliers, standard financial data
# rolling window scaling:
#   - no future data leakage
#   - adapt to market conditions changes 
#   - more realistic for real-time trading
