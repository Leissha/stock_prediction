import numpy as np

def create_sequences(scaled_data, lag_days, lookup_step, target_columns_names, all_column_names):
    """
    Create sequences for LSTM training with future prediction capability
    
    Args:
        scaled_data (np.array): Scaled feature data (n_samples, n_features)
        lag_days (int): Number of historical days to use as input
        lookup_step (int): Steps into future to predict (1=next day, 5=next week)
        target_columns_names (list): Names of columns to predict
        all_column_names (list): Names of all columns in scaled_data
        
    Returns:
        tuple: (X_sequences, y_sequences)
            X: (n_samples, lag_days, n_features) - Historical sequences
            y: (n_samples, n_targets) - Future targets
            
    Sequence creation logic:
    - For each valid position i in the data:
      - X[sample] = data[i-lag_days:i] (past lag_days of all features)
      - y[sample] = data[i+lookup_step, target_indices] (future target values)
    - This ensures we predict future values using only past information
    """
    # Convert target column names to array indices
    target_indices = [all_column_names.index(col) for col in target_columns_names]
    print(f"Target indices: {target_indices} for columns: {target_columns_names}")
    
    X, y = [], []
    
    # Window bounds:
    # start at i = lag_days so there is enough history for the first window
    # stop at len(scaled_data) - lookup_step + 1 so i+lookup_step-1 is in-bounds
    for i in range(lag_days, len(scaled_data) - lookup_step + 1):
        # Historical sequence: all features for past lag_days
        X.append(scaled_data[i-lag_days:i, :])
        
        # Future target: only target features at future timestep
        y.append(scaled_data[i + lookup_step - 1, target_indices])
        
    X = np.array(X)  # Shape: (n_samples, lag_days, n_features)
    y = np.array(y)  # Shape: (n_samples, n_targets)
    
    print(f"Created {len(X)} sequences with lag_days={lag_days}, lookup_step={lookup_step}")
    return X, y