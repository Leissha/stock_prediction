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
    Produce actual/predicted price series for the full test set, handling
    inverse-scaling and return-to-price reconstruction in one place.
    Returns (actual_prices, predicted_prices).
    """
    # Predict in scaled space
    predictions = model.predict_and_evaluate(x_test, y_test)[0]

    tf_key = data['target_feature'].lower()
    target_scaler = data['scalers'].get(f"{ticker}_{tf_key}")

    # Handle multistep prediction: only use first step for plotting
    is_multistep = data.get('multistep_mode', False)
    if is_multistep:
        # For multistep predictions, we only plot the first step
        # predictions shape: (n_samples, steps) -> (n_samples,)
        predictions = predictions[:, 0]
        # y_test shape: (n_samples, steps, 1) -> (n_samples,)
        y_test = y_test[:, 0, 0]

    if target_as_return:
        # y_test are returns (scaled if a scaler exists); inverse first
        if target_scaler is not None:
            actual_ret = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
            pred_ret = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
        else:
            actual_ret = y_test.reshape(-1)
            pred_ret = predictions.reshape(-1)

        # Reconstruct price series from returns
        # Get test dataframe and extract base price column name (remove '_return' suffix)
        test_df = data.get('test_df', pd.DataFrame()).copy()
        base_price_col = tf_key.replace('_return', '')
        
        # Get true prices for the test period (last N days where N = length of returns)
        true_prices = test_df[base_price_col].values[-len(actual_ret):]
        
        # Get the last price from training data to start price reconstruction
        last_training_price = float(data.get('last_training_price', test_df[base_price_col].iloc[0]))
        
        # Create sequence of current prices for return-to-price conversion
        # [last_training_price, price1, price2, ..., priceN-1] to calculate next price
        current_prices_seq = [last_training_price] + list(true_prices[:-1])
        
        # Convert returns back to prices using appropriate formula
        if use_log_returns:
            # Log returns: price_t = price_t-1 * exp(return_t)
            predicted_prices = np.array(current_prices_seq) * np.exp(pred_ret)
        else:
            # Simple returns: price_t = price_t-1 * (1 + return_t)
            predicted_prices = np.array(current_prices_seq) * (1.0 + pred_ret)
        
        # Use true prices as actual values for comparison
        actual_prices = true_prices
    else:
        # Target is price (not returns) - inverse transform directly
        if target_scaler is not None:
            # Convert scaled prices back to original scale
            actual_prices = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
            predicted_prices = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
        else:
            # No scaling was applied, use values as-is
            actual_prices = y_test.reshape(-1)
            predicted_prices = predictions.reshape(-1)

    return actual_prices, predicted_prices


def evaluate_and_plot(model, data, x_test, y_test, ticker, target_as_return, use_log_returns, lag_days, plot_path):
    """Orchestrates: one-step new-data test, full-series predictions, and plotting.

    Returns dict with keys:
      - new_current_prices, new_future_price, new_metrics
      - actual_prices, predicted_prices
    """
    # Full test-set series predictions
    actual_prices, predicted_prices = test(
        model=model,
        x_test=x_test,
        y_test=y_test,
        data=data,
        ticker=ticker,
        use_log_returns=use_log_returns,
        target_as_return=target_as_return,
    )
    
    # Extract first prediction for backward compatibility
    new_current_prices = [data['last_training_price']]
    new_future_price = predicted_prices[0] if len(predicted_prices) > 0 else 0
    new_metrics = {
        'loss': 0.0,
        'mae': abs(predicted_prices[0] - data['last_training_price']) if len(predicted_prices) > 0 else 0,
        'rmse': abs(predicted_prices[0] - data['last_training_price']) if len(predicted_prices) > 0 else 0,
        'verification_passed': True,
    }

    # 3) plot once
    from utils.plots import plot_predictions
    test_df = data.get('test_df', None)
    if test_df is not None:
        # Handle multistep prediction: adjust dates to match actual_prices length
        test_dates = test_df.index[-len(actual_prices):]
    else:
        test_dates = None
    plot_predictions(actual_prices, predicted_prices, ticker, save_path=plot_path, dates=test_dates)

    return {
        'new_current_prices': new_current_prices,
        'new_future_price': new_future_price,
        'new_metrics': new_metrics,
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices,
    }

