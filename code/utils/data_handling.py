import numpy as np


def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()  # Save current random state
    np.random.shuffle(a)           # Shuffle first array
    np.random.set_state(state)     # Restore same random state
    np.random.shuffle(b)           # Shuffle second array with same pattern
    


def create_sliding_window(scaled_data, original_data, prediction_days):
    """
    Create a sliding window for training data
    Args:
        scaled_data: The scaled data for input sequences
        original_data: The original unscaled data for target values
        prediction_days: The number of days to predict
    Returns:
        x_train: The training data (scaled)
        y_train: The target data (original scale)
    """
    x_train = []
    y_train = []
    
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x])    # 1D slice (scaled)
        # Target is the CURRENT value from the original unscaled series (predicting current day from past days)
        y_train.append(original_data.iloc[x])
    
    # Convert to arrays
    x_train = np.array(x_train)  # 2D array (samples, time_steps)
    y_train = np.array(y_train)  # 1D array (samples, )
    
    # Reshape for LSTM 3D input req: (samples, time_steps, features):
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train

#------------------------------------------------------------------------------
# Example:
#------------------------------------------------------------------------------
# Input: 1D array
# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     ↓
# Sliding Window: 2D array
# [[0.1, 0.2, 0.3],
#  [0.2, 0.3, 0.4],
#  [0.3, 0.4, 0.5],
#  [0.4, 0.5, 0.6],
#  [0.5, 0.6, 0.7],
#  [0.6, 0.7, 0.8],
#  [0.7, 0.8, 0.9]]
#     ↓
# Reshape: 3D array
# [[[0.1], [0.2], [0.3]],
#  [[0.2], [0.3], [0.4]],
#  [[0.3], [0.4], [0.5]],
#  [[0.4], [0.5], [0.6]],
#  [[0.5], [0.6], [0.7]],
#  [[0.6], [0.7], [0.8]],
#  [[0.7], [0.8], [0.9]]]



# gonna apply z-score normalization and rolling window scaling
# to the data
# z-score normalization:
#   - handle data leakage
#   - robust to outliers, standard financial data
# rolling window scaling:
#   - no future data leakage
#   - adapt to market conditions changes 
#   - more realistic for real-time trading
