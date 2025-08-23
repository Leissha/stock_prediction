from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from loguru import logger
    
def calculate_trading_metrics(actual_prices, predicted_prices, current_prices=None, lookup_step=1, 
                            future_price=None, loss_value=None, loss_name="loss", filename=None, scale=True,
                            feature_names=None, evaluation_mode="price"):
    """
    Calculate trading-based accuracy metrics and optionally print/save results.
    
    Args:
        actual_prices: Array of actual future prices/features (can be 1D or 2D)
        predicted_prices: Array of predicted future prices/features (can be 1D or 2D)
        current_prices: Array of current prices (if None, will use actual_prices as baseline)
        lookup_step: Number of days into the future being predicted (default=1 for next day)
        future_price: Predicted future price for output (optional)
        loss_value: Model loss value for output (optional)
        loss_name: Name of the loss function (default="loss")
        filename: Output CSV filename (optional, if provided will save results)
        scale: Whether the data is scaled (default=True, affects MAE calculation)
        feature_names: List of feature names for multi-feature evaluation (optional)
        evaluation_mode: Evaluation mode - "price" (trading metrics) or "general" (basic metrics only)
    
    Returns:
        dict: Dictionary containing all accuracy metrics
    """
    # Handle multi-feature vs single-feature data
    actual_prices = np.array(actual_prices)
    predicted_prices = np.array(predicted_prices)
    
    # Determine if we're dealing with multi-feature data
    is_multi_feature = actual_prices.ndim > 1 and actual_prices.shape[1] > 1
    
    if is_multi_feature:
        # Multi-feature case
        n_features = actual_prices.shape[1]
        n_samples = actual_prices.shape[0]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError(f"Number of feature names ({len(feature_names)}) doesn't match number of features ({n_features})")
        
        # For multi-feature, we'll evaluate each feature separately
        all_metrics = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_actual = actual_prices[:, i]
            feature_predicted = predicted_prices[:, i]
            
            # For multi-feature, use general evaluation mode unless it's a price feature
            feature_mode = "price" if "price" in feature_name.lower() or "close" in feature_name.lower() else "general"
            
            feature_metrics = _calculate_single_feature_metrics(
                feature_actual, feature_predicted, current_prices, lookup_step,
                future_price, loss_value, loss_name, scale, feature_mode
            )
            all_metrics[feature_name] = feature_metrics
        
        # Aggregate metrics across features
        metrics = _aggregate_multi_feature_metrics(all_metrics, feature_names)
        
    else:
        # Single-feature case (original logic)
        if current_prices is None:
            current_prices = actual_prices
        
        # Ensure all arrays are 1D and have same length
        actual_prices = actual_prices.flatten()
        predicted_prices = predicted_prices.flatten()
        current_prices = np.array(current_prices).flatten()
        
        if len(actual_prices) != len(predicted_prices) or len(actual_prices) != len(current_prices):
            raise ValueError("All price arrays must have the same length")
        
        metrics = _calculate_single_feature_metrics(
            actual_prices, predicted_prices, current_prices, lookup_step,
            future_price, loss_value, loss_name, scale, evaluation_mode
        )
    
    # Print and save results if requested
    if future_price is not None and loss_value is not None:
        _print_and_save_results(metrics, future_price, loss_value, loss_name, lookup_step, filename)
    
    return metrics


def _calculate_single_feature_metrics(actual_prices, predicted_prices, current_prices, lookup_step,
                                    future_price, loss_value, loss_name, scale, evaluation_mode):
    """Helper function to calculate metrics for a single feature"""
    
    # Define profit calculation functions (same as p1)
    buy_profit = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    
    # Calculate profits for each prediction (only for price mode)
    buy_profits = []
    sell_profits = []
    
    if evaluation_mode == "price":
        for i in range(len(actual_prices)):
            current = current_prices[i]
            pred_future = predicted_prices[i]
            true_future = actual_prices[i]
            
            buy_profits.append(buy_profit(current, pred_future, true_future))
            sell_profits.append(sell_profit(current, pred_future, true_future))
        
        # Calculate trading metrics
        total_buy_profit = sum(buy_profits)
        total_sell_profit = sum(sell_profits)
        total_profit = total_buy_profit + total_sell_profit
        profit_per_trade = total_profit / len(actual_prices) if len(actual_prices) > 0 else 0
        
        # Calculate accuracy score (percentage of profitable trades)
        profitable_trades = sum(1 for bp, sp in zip(buy_profits, sell_profits) if bp > 0 or sp > 0)
        accuracy_score = profitable_trades / len(actual_prices) if len(actual_prices) > 0 else 0
    else:
        # General mode - basic metrics only
        total_buy_profit = total_sell_profit = total_profit = profit_per_trade = 0
        profitable_trades = 0
        accuracy_score = 0
        buy_profits = sell_profits = []
    
    # Calculate Mean Absolute Error
    if scale:
        # If data is scaled, MAE is already in original scale (after inverse transform)
        mae = np.mean(np.abs(actual_prices - predicted_prices))
    else:
        # If data is not scaled, use the loss_value if provided (from model.evaluate)
        if loss_value is not None:
            mae = loss_value
        else:
            # Fallback to calculating MAE directly
            mae = np.mean(np.abs(actual_prices - predicted_prices))
    
    return {
        'accuracy_score': accuracy_score,
        'total_buy_profit': total_buy_profit,
        'total_sell_profit': total_sell_profit,
        'total_profit': total_profit,
        'profit_per_trade': profit_per_trade,
        'mean_absolute_error': mae,
        'profitable_trades': profitable_trades,
        'total_trades': len(actual_prices),
        'buy_profits': buy_profits,
        'sell_profits': sell_profits
    }


def _aggregate_multi_feature_metrics(all_metrics, feature_names):
    """Helper function to aggregate metrics across multiple features"""
    
    # Aggregate basic metrics
    total_mae = sum(metrics['mean_absolute_error'] for metrics in all_metrics.values())
    avg_mae = total_mae / len(feature_names)
    
    # Aggregate trading metrics (only for price features)
    price_features = [name for name in feature_names if "price" in name.lower() or "close" in name.lower()]
    
    if price_features:
        total_profit = sum(all_metrics[name]['total_profit'] for name in price_features)
        total_trades = sum(all_metrics[name]['total_trades'] for name in price_features)
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
    else:
        total_profit = avg_profit_per_trade = 0
    
    return {
        'mean_absolute_error': avg_mae,
        'total_profit': total_profit,
        'profit_per_trade': avg_profit_per_trade,
        'feature_metrics': all_metrics,
        'feature_names': feature_names
    }


def _print_and_save_results(metrics, future_price, loss_value, loss_name, lookup_step, filename):
    """Helper function to print and save results"""
    
    print(f"Future price after {lookup_step} day{'s' if lookup_step > 1 else ''} is {future_price:.2f}$")
    print(f"{loss_name}: {loss_value}")
    print(f"Mean Absolute Error: {metrics['mean_absolute_error']}")
    
    if 'accuracy_score' in metrics:
        print(f"Accuracy score: {metrics['accuracy_score']}")
        print(f"Total buy profit: {metrics['total_buy_profit']}")
        print(f"Total sell profit: {metrics['total_sell_profit']}")
        print(f"Total profit: {metrics['total_profit']}")
        print(f"Profit per trade: {metrics['profit_per_trade']}")
    
    if filename:
        with open(filename, 'w') as f:
            f.write(f"Future price after {lookup_step} day{'s' if lookup_step > 1 else ''} is {future_price:.2f}$\n")
            f.write(f"{loss_name}: {loss_value}\n")
            f.write(f"Mean Absolute Error: {metrics['mean_absolute_error']}\n")
            
            if 'accuracy_score' in metrics:
                f.write(f"Accuracy score: {metrics['accuracy_score']}\n")
                f.write(f"Total buy profit: {metrics['total_buy_profit']}\n")
                f.write(f"Total sell profit: {metrics['total_sell_profit']}\n")
                f.write(f"Total profit: {metrics['total_profit']}\n")
                f.write(f"Profit per trade: {metrics['profit_per_trade']}\n")


