import numpy as np

def create_sliding_window(scaled_data, prediction_days):
    """
    Create a sliding window for training data
    Args:
        scaled_data: The scaled data
        prediction_days: The number of days to predict
    Returns:
        x_train: The training data
        y_train: The target data
    """
    x_train = []
    y_train = []
    
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x])    # 1D slice
        y_train.append(scaled_data[x])                      # 1D single value
    
    # Convert to arrays
    x_train = np.array(x_train)  # 2D array (samples, time_steps)
    y_train = np.array(y_train)  # 1D array (samples, )
    
    # Resape for LSTM 3D input req: (samples, time_steps, features):
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
