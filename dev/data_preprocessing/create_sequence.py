import numpy as np

def create_sequences(scaled_data, lag_days, lookup_steps, target_columns_names, all_column_names):
    """
    Create sequences for LSTM training with future prediction capability
    
    Args:
        scaled_data (np.array): Scaled feature data (n_samples, n_features)
        lag_days (int): Number of historical days to use as input
        lookup_steps (int): Number of future days to predict (1=single-step, >1=multistep)
        target_columns_names (list): Names of columns to predict
        all_column_names (list): Names of all columns in scaled_data
        
    Returns:
        tuple: (X_sequences, y_sequences)
            X: (n_samples, lag_days, n_features) - Historical sequences
            y: (n_samples, 1) or (n_samples, lookup_steps, 1) - Future targets
            
    Sequence creation logic:
    - For each valid position i in the data:
      - X[sample] = data[i-lag_days:i] (past lag_days of all features)
      - y[sample] = data[i+step, target_indices] for step in range(lookup_steps)
    - This ensures we predict future values using only past information
    """
    # Convert target column names to array indices
    target_indices = [all_column_names.index(col) for col in target_columns_names]
    print(f"Target indices: {target_indices} for columns: {target_columns_names}")
    
    X, y = [], []
    
    # Window bounds:
    # start at i = lag_days so there is enough history for the first window
    # stop at len(scaled_data) - lookup_steps to allow indexing i + step + 1 (t+1..t+k)
    for i in range(lag_days, len(scaled_data) - lookup_steps):
        # Historical sequence: all features for past lag_days
        X.append(scaled_data[i-lag_days:i, :])
        
        # Future target: create sequence of future values (t+1 .. t+k)
        future_sequence = []
        for step in range(lookup_steps):
            future_sequence.append(scaled_data[i + step + 1, target_indices])
        
        # Always return as array for consistency
        y.append(future_sequence)
        
    X = np.array(X)  # Shape: (n_samples, lag_days, n_features)
    y = np.array(y)  # Shape: (n_samples, n_targets) or (n_samples, lookup_steps, n_targets)
    

    print(f"Created {len(X)} sequences: X{X.shape} → y{y.shape}")
    return X, y
