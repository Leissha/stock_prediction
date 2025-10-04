import numpy as np
import pandas as pd
from typing import List, Tuple


def predict_and_descale(model, x_test, y_test, target_scaler=None):
    """
    Simplified prediction function with single-scaler inverse transform
    
    Args:
        model: Trained model
        x_test: Test input data
        y_test: Test target data (scaled)
        data: Dictionary containing scalers and metadata from DataProcessor

    Returns:
        tuple: (actual_prices, predicted_prices, metrics) - all in original scale
    """
    # Make predictions using model
    predictions, metrics = model.predict_and_evaluate(x_test, y_test)
    
    # Inverse transform using separate scaler for target feature
    if target_scaler is not None:
        actual_prices = target_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).reshape(-1)
        predicted_prices = target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(-1)
        
        print(f"Inverse transformed using scaler: {target_scaler}")
    else:
        # No scaling was applied, use values as-is
        actual_prices = y_test.reshape(-1)
        predicted_prices = predictions.reshape(-1)
        print("No inverse transform applied - values used as-is")
        
    return actual_prices, predicted_prices, metrics


def test(model, x_test, y_test, data, ticker, use_log_returns: bool, target_as_return: bool):
    """
    Generate multistep predictions and reconstruct price sequences for evaluation.
    
    Logic:
    1. Get raw model predictions (scaled)
    2. Build true price sequence y[t+1..t+k] from test data
    3. For return targets: compound predicted returns to get price paths
    4. For price targets: inverse-transform predicted prices directly
    5. Return both full sequences and last-step prices for evaluation
    
    Returns:
        tuple: (actual_prices_final, predicted_prices_final, y_seq, y_hat_seq)
        - actual_prices_final: true prices at final horizon t+k, shape (N,)
        - predicted_prices_final: predicted prices at final horizon t+k, shape (N,)
        - y_seq: true prices for all horizons t+1..t+k, shape (N, k)
        - y_hat_seq: predicted prices for all horizons t+1..t+k, shape (N, k)
    """
    # Get target feature name and corresponding scaler for inverse transformation
    target_feature_name = data['target_feature'].lower()
    target_feature_scaler = data['scalers'].get(f"{ticker}_{target_feature_name}")
    
    # Get raw model predictions (still scaled)
    raw_scaled_predictions = model.predict_and_evaluate(x_test, y_test)[0]  # Shape: (N, k) or (N, 1)
    
    # Extract prediction horizon length
    lookup_steps = int(data.get('lookup_steps', 1))
    
    # Get test data and build true price sequence
    test_dataframe = data.get('test_df', pd.DataFrame())
    target_price_col = target_feature_name.replace('_return', '')  # Remove '_return' suffix if present
    y_true_df = test_dataframe[target_price_col].to_numpy()
    
    # Calculate number of valid sequences (accounting for horizon)
    # E.g. If we have 100 days of test data and want to predict 5 days ahead:
    #       We can only create sequences that start at day 1 and end at day 95
    #       Because sequence starting at day 96 would need days 97, 98, 99, 100, 101 (but day 101 doesn't exist)
    num_valid_sequences = max(0, len(y_true_df) - lookup_steps) # ensure no negative values
    if num_valid_sequences == 0:
        return np.array([]), np.array([]), None, None
    
    # Get starting prices y[t] at time t to create future sequences
    y_t = y_true_df[:num_valid_sequences]
    
    # Build true price sequence: y[t+1], y[t+2], ..., y[t+k]
    # Each column h represents prices at horizon h (t+h)
    
    #   Original data: [100, 102, 105, 103, 101]
    #   Prediction horizon: 3 days

    #   Sequences we can create:
    #   Sequence 0: base=100, future=[102, 105, 103]
    #   Sequence 1: base=102, future=[105, 103, 101]  

    #   y_seq matrix:
    #   [[102, 105, 103],    # Sequence 0: prices at t+1, t+2, t+3
    #    [105, 103, 101]]    # Sequence 1: prices at t+1, t+2, t+3

    y_seq = np.column_stack([
        y_true_df[horizon:horizon + num_valid_sequences] 
        for horizon in range(1, lookup_steps + 1)
    ])  # create matrix of shape: (N_sequences, lookup_steps)
    

    if target_as_return:
        # CASE 1: Model predicts returns, need to compound to get prices
        y_hat_returns = raw_scaled_predictions  # Predicted returns sequence (N, k)
        
        # Inverse-transform predicted returns if scaler exists
        if target_feature_scaler is not None:
            flattened_returns = y_hat_returns.reshape(-1, 1)
            inverse_transformed_returns = target_feature_scaler.inverse_transform(flattened_returns)
            y_hat_returns = inverse_transformed_returns.reshape(y_hat_returns.shape)
        
        # Compound returns to get price paths
        if use_log_returns:
            # For log returns: y[t+k] = y[t] * exp(sum(r[t+1] + r[t+2] + ... + r[t+k]))
            # https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
            cumulative_log_return_factors = np.exp(np.cumsum(y_hat_returns, axis=1))
        else:
            # For simple returns: y[t+k] = y[t] * (1+r[t+1]) * (1+r[t+2]) * ... * (1+r[t+k])
            # https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html
            cumulative_return_factors = np.cumprod(1.0 + y_hat_returns, axis=1)
            cumulative_log_return_factors = cumulative_return_factors
        
        # Apply compounding: multiply base prices by cumulative factors
        Y_hat = y_t[:, None] * cumulative_log_return_factors  # Predicted prices sequence (N, k)
        
    else:
        # CASE 2: Model predicts prices directly
        Y_hat = raw_scaled_predictions  # Predicted prices sequence (N, k)
        
        # Inverse-transform predicted prices if scaler exists
        if target_feature_scaler is not None:
            flattened_prices = Y_hat.reshape(-1, 1)
            inverse_transformed_prices = target_feature_scaler.inverse_transform(flattened_prices)
            Y_hat = inverse_transformed_prices.reshape(Y_hat.shape)
        
        # Ensure we only take valid sequences
        Y_hat = Y_hat[:num_valid_sequences, :]

    # Extract final horizon prices for evaluation (t+k)
    y_k = y_seq[:, -1]  # True prices at t+k
    y_hat_k = Y_hat[:, -1]  # Predicted prices at t+k
    
    return (y_k, y_hat_k, y_seq, Y_hat)


def test_and_evaluate(model, data, x_test, y_test, ticker, target_as_return, use_log_returns, lag_days, plot_path):
    """
    Orchestrate multistep prediction testing, evaluation, and visualization.
    
    Logic:
    1. Generate predictions for all test sequences
    2. Calculate MAE/RMSE on final horizon prices
    3. Prepare data for plotting (training context + test predictions)
    4. Generate per-horizon metrics for analysis
    
    Returns:
        dict: Contains prediction results, metrics, and sequences for analysis
    """
    # Full test-set series predictions
    actual_prices, predicted_prices, true_seq, pred_seq = test(
        model=model,
        x_test=x_test,
        y_test=y_test,
        data=data,
        ticker=ticker,
        use_log_returns=use_log_returns,
        target_as_return=target_as_return,
    )
    
    # Calculate evaluation metrics on final horizon (t+k)
    import numpy as np
    mae = float(np.mean(np.abs(actual_prices - predicted_prices))) if len(actual_prices) else 0.0
    rmse = float(np.sqrt(np.mean((actual_prices - predicted_prices)**2))) if len(actual_prices) else 0.0
    
    # Extract first prediction for backward compatibility
    current_prices = [data['last_training_price']]
    future_price = predicted_prices[0] if len(predicted_prices) > 0 else 0
    metrics = {
        'loss': 0.0,
        'mae': mae,
        'rmse': rmse,
        'verification_passed': True,
    }

    # Generate prediction visualization plot
    from utils.plots import plot_predictions
    test_df = data.get('test_df', None)
    if test_df is not None:
        # Handle multistep prediction: adjust dates to match actual_prices length
        test_dates = test_df.index[-len(actual_prices):]
    else:
        test_dates = None
    
    # Prepare training period data for context
    train_df = data.get('train_df', None)
    if train_df is not None:
        base_col = data['target_feature'].replace('_return','')
        train_dates = train_df.index[-min(len(train_df), lag_days):]
        train_prices = train_df[base_col].iloc[-len(train_dates):].values
        print(f"Training data: {len(train_dates)} points, {len(train_prices)} prices")
    else:
        train_dates, train_prices = None, None
        print("No training data available")
    
    # Create prediction plot
    plot_predictions(actual_prices, predicted_prices, ticker, 
                    save_path=plot_path, dates=test_dates, 
                    train_dates=train_dates, train_prices=train_prices)

    # Calculate per-horizon metrics for detailed analysis
    per_horizon_metrics = []
    if (true_seq is not None) and (pred_seq is not None):
        target_price_col = data['target_feature'].replace('_return','')
        y_true_df = data['test_df'][target_price_col].to_numpy()
        num_sequences = true_seq.shape[0]
        base_prices_reference = y_true_df[:num_sequences]
        prediction_horizon = true_seq.shape[1]
        
        # Calculate metrics for each horizon step
        for horizon_step in range(prediction_horizon):
            true_prices_at_horizon = true_seq[:, horizon_step]
            predicted_prices_at_horizon = pred_seq[:, horizon_step]
            
            # MAE and RMSE for this horizon
            mae_at_horizon = float(np.mean(np.abs(true_prices_at_horizon - predicted_prices_at_horizon)))
            rmse_at_horizon = float(np.sqrt(np.mean((true_prices_at_horizon - predicted_prices_at_horizon)**2)))
            
            # Pure directional accuracy: sign(y[t+h] - y[t]) vs sign(ŷ[t+h] - y[t])
            true_direction_signs = np.sign(true_prices_at_horizon - base_prices_reference)
            predicted_direction_signs = np.sign(predicted_prices_at_horizon - base_prices_reference)
            
            # Only count non-zero directions (exclude flat movements)
            valid_direction_mask = (true_direction_signs != 0) & (predicted_direction_signs != 0)
            pure_directional_accuracy = float(np.mean(true_direction_signs[valid_direction_mask] == predicted_direction_signs[valid_direction_mask])) if valid_direction_mask.any() else 0.0
            
            per_horizon_metrics.append({
                "horizon": horizon_step + 1, 
                "mae": mae_at_horizon, 
                "rmse": rmse_at_horizon, 
                "pure_directional_accuracy": pure_directional_accuracy
            })

    # Calculate k-step directional accuracy and baselines
    if len(actual_prices) > 0 and len(predicted_prices) > 0:
        prediction_horizon = data.get('lookup_steps', 1)
        target_price_col = data['target_feature'].replace('_return','')
        full_test_price_series = data['test_df'][target_price_col].to_numpy()
        
        # Calculate valid sequences for k-step evaluation
        num_valid_sequences = len(full_test_price_series) - prediction_horizon
        y_t = full_test_price_series[:num_valid_sequences]  # Starting prices y[t]
        y_k = full_test_price_series[prediction_horizon:]  # Final prices y[t+k]
        y_hat_k = predicted_prices[:num_valid_sequences]  # Predicted y[t+k]

        # Calculate directional accuracy: did we predict the right direction?
        # Compare: sign(y[t+k] - y[t]) vs sign(ŷ[t+k] - y[t])
        true_direction_signs = np.sign(y_k - y_t)  # True direction: up/down/flat
        predicted_direction_signs = np.sign(y_hat_k - y_t)  # Predicted direction
        
        # Only count non-flat movements (exclude cases where price didn't change)
        valid_direction_mask = (true_direction_signs != 0) & (predicted_direction_signs != 0)
        pure_directional_accuracy = float(np.mean(true_direction_signs[valid_direction_mask] == predicted_direction_signs[valid_direction_mask])) if valid_direction_mask.any() else 0.0

        # Calculate baselines: what if we just guessed randomly?
        # Baseline "always up": percentage of time prices actually went up
        baseline_always_up = float(np.mean(true_direction_signs > 0))
        # Baseline "always down": percentage of time prices actually went down  
        baseline_always_down = float(np.mean(true_direction_signs < 0))

        print(f"[k={prediction_horizon}] Pure DA={pure_directional_accuracy:.4f} | Baseline up={baseline_always_up:.4f} down={baseline_always_down:.4f} | N_used={int(valid_direction_mask.sum())}")
        
        # Add to metrics
        metrics.update({
            'pure_directional_accuracy': pure_directional_accuracy,
            'baseline_up': baseline_always_up,
            'baseline_down': baseline_always_down,
            'n_used': int(valid_direction_mask.sum())
        })

    return {
        'current_prices': current_prices,
        'future_price': future_price,
        'metrics': metrics,
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices,
        'actual_seq': true_seq,
        'pred_seq': pred_seq,
        'per_h_metrics': per_horizon_metrics,
    }

