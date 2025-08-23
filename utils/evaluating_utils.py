"""
Trading Evaluation Utilities

This module provides functions to evaluate machine learning models for financial trading applications.
It calculates both traditional ML metrics (MAE) and trading-specific metrics (profit, accuracy).

Key Features:
- Single and multi-feature evaluation
- Trading simulation with buy/sell decisions
- Profit calculation and trading accuracy
- Flexible output options (print, save to file)

Author: [Your Name]
Date: [Current Date]
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, List, Dict, Union, Tuple, Any


def calculate_trading_metrics(
    actual_prices: Union[np.ndarray, List],
    predicted_prices: Union[np.ndarray, List],
    current_prices: Optional[Union[np.ndarray, List]] = None,
    lookup_step: int = 1,
    future_price: Optional[float] = None,
    loss_value: Optional[float] = None,
    loss_name: str = "loss",
    filename: Optional[str] = None,
    scale: bool = True,
    feature_names: Optional[List[str]] = None,
    evaluation_mode: str = "price"
) -> Dict[str, Any]:
    """
    Calculate comprehensive trading metrics for ML model evaluation.
    
    This function evaluates machine learning predictions for financial trading by calculating:
    1. Traditional ML metrics (Mean Absolute Error)
    2. Trading-specific metrics (profit, trading accuracy)
    3. Multi-feature support for complex models
    
    Args:
        actual_prices: True values from test data (1D for single feature, 2D for multiple features)
        predicted_prices: Model predictions (same shape as actual_prices)
        current_prices: Base prices for profit calculation (defaults to actual_prices if None)
        lookup_step: Prediction horizon in days (e.g., 1 = next day, 7 = next week)
        future_price: Single prediction value for display purposes
        loss_value: Model's loss metric for reporting
        loss_name: Name of the loss function used
        filename: Path to save results as text file
        scale: Whether input data is scaled (affects MAE calculation)
        feature_names: Names for each feature in multi-feature data
        evaluation_mode: "price" for trading metrics, "general" for basic metrics only
    
    Returns:
        Dictionary containing:
        - accuracy_score: Percentage of profitable trades
        - total_profit: Sum of all trading profits
        - profit_per_trade: Average profit per trade
        - mean_absolute_error: Average prediction error
        - For multi-feature: feature_metrics with per-feature breakdown
    
    Example:
        >>> actual = [100, 102, 98, 105]
        >>> predicted = [101, 103, 97, 104]
        >>> metrics = calculate_trading_metrics(actual, predicted)
        >>> print(f"Trading accuracy: {metrics['accuracy_score']:.2%}")
    """
    # Convert inputs to numpy arrays for consistent processing
    actual_prices = np.array(actual_prices)
    predicted_prices = np.array(predicted_prices)
    
    # Check if we're dealing with multi-feature data (2D array with multiple columns)
    is_multi_feature = actual_prices.ndim > 1 and actual_prices.shape[1] > 1
    
    if is_multi_feature:
        # === MULTI-FEATURE EVALUATION ===
        # When dealing with multiple features (e.g., Open, High, Low, Close, Volume)
        n_features = actual_prices.shape[1]
        n_samples = actual_prices.shape[0]
        
        # Generate default feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) doesn't match "
                f"number of features ({n_features})"
            )
        
        # Evaluate each feature separately
        all_metrics = {}
        logger.info(f"Evaluating {n_features} features: {feature_names}")
        
        for i, feature_name in enumerate(feature_names):
            feature_actual = actual_prices[:, i]
            feature_predicted = predicted_prices[:, i]
            
            # Determine evaluation mode based on feature name
            # Only price-related features get trading metrics
            feature_mode = (
                "price" if any(keyword in feature_name.lower() 
                              for keyword in ["price", "close", "open", "high", "low"])
                else "general"
            )
            
            logger.debug(f"Evaluating {feature_name} in {feature_mode} mode")
            
            feature_metrics = _calculate_single_feature_metrics(
                feature_actual, feature_predicted, current_prices, lookup_step,
                future_price, loss_value, loss_name, scale, feature_mode
            )
            all_metrics[feature_name] = feature_metrics
        
        # Combine metrics from all features
        metrics = _aggregate_multi_feature_metrics(all_metrics, feature_names)
        
    else:
        # === SINGLE-FEATURE EVALUATION ===
        # Traditional single-feature evaluation (e.g., just Close price)
        
        # Use actual prices as current prices if not provided
        if current_prices is None:
            current_prices = actual_prices
            logger.debug("Using actual prices as current prices for profit calculation")
        
        # Ensure all arrays are 1D and have consistent length
        actual_prices = actual_prices.flatten()
        predicted_prices = predicted_prices.flatten()
        current_prices = np.array(current_prices).flatten()
        
        # Validate array lengths
        if len(actual_prices) != len(predicted_prices):
            raise ValueError(
                f"Actual prices length ({len(actual_prices)}) doesn't match "
                f"predicted prices length ({len(predicted_prices)})"
            )
        if len(actual_prices) != len(current_prices):
            raise ValueError(
                f"Actual prices length ({len(actual_prices)}) doesn't match "
                f"current prices length ({len(current_prices)})"
            )
        
        logger.info(f"Evaluating single feature with {len(actual_prices)} samples")
        
        metrics = _calculate_single_feature_metrics(
            actual_prices, predicted_prices, current_prices, lookup_step,
            future_price, loss_value, loss_name, scale, evaluation_mode
        )
    
    # Output results if requested
    if future_price is not None and loss_value is not None:
        _print_and_save_results(metrics, future_price, loss_value, loss_name, lookup_step, filename)
    
    return metrics


def _calculate_single_feature_metrics(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    current_prices: np.ndarray,
    lookup_step: int,
    future_price: Optional[float],
    loss_value: Optional[float],
    loss_name: str,
    scale: bool,
    evaluation_mode: str
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for a single feature (e.g., Close price).
    
    This function implements a simple trading strategy:
    - BUY when model predicts price will go up (pred_future > current)
    - SELL when model predicts price will go down (pred_future < current)
    - Calculate profit/loss based on actual outcomes
    
    Args:
        actual_prices: True values for this feature
        predicted_prices: Model predictions for this feature
        current_prices: Base prices for profit calculation
        lookup_step: Prediction horizon (not used in current implementation)
        future_price: Single prediction for display (not used in calculation)
        loss_value: Model loss for display purposes
        loss_name: Name of loss function
        scale: Whether data is scaled (affects MAE calculation)
        evaluation_mode: "price" for trading metrics, "general" for basic metrics only
    
    Returns:
        Dictionary with calculated metrics
    """
    
    # === TRADING STRATEGY FUNCTIONS ===
    # Buy if we predict price will increase, profit = actual_increase
    def calculate_buy_profit(current: float, predicted_future: float, actual_future: float) -> float:
        """Calculate profit from a buy decision"""
        if predicted_future > current:  # Model predicts price increase -> BUY
            return actual_future - current  # Profit = actual price change
        return 0  # No trade made
    
    # Sell if we predict price will decrease, profit = actual_decrease  
    def calculate_sell_profit(current: float, predicted_future: float, actual_future: float) -> float:
        """Calculate profit from a sell decision"""
        if predicted_future < current:  # Model predicts price decrease -> SELL
            return current - actual_future  # Profit = negative of actual price change
        return 0  # No trade made
    
    # Initialize profit tracking
    buy_profits = []
    sell_profits = []
    
    if evaluation_mode == "price":
        # === TRADING SIMULATION ===
        logger.debug(f"Running trading simulation on {len(actual_prices)} samples")
        
        for i in range(len(actual_prices)):
            current = current_prices[i]
            pred_future = predicted_prices[i]
            true_future = actual_prices[i]
            
            # Calculate profits for both strategies
            buy_profit = calculate_buy_profit(current, pred_future, true_future)
            sell_profit = calculate_sell_profit(current, pred_future, true_future)
            
            buy_profits.append(buy_profit)
            sell_profits.append(sell_profit)
        
        # === AGGREGATE TRADING METRICS ===
        total_buy_profit = sum(buy_profits)
        total_sell_profit = sum(sell_profits)
        total_profit = total_buy_profit + total_sell_profit
        profit_per_trade = total_profit / len(actual_prices) if len(actual_prices) > 0 else 0
        
        # Calculate trading accuracy (percentage of profitable trades)
        profitable_trades = sum(
            1 for bp, sp in zip(buy_profits, sell_profits) 
            if bp > 0 or sp > 0  # Either buy or sell was profitable
        )
        accuracy_score = profitable_trades / len(actual_prices) if len(actual_prices) > 0 else 0
        
        logger.debug(
            f"Trading results: {profitable_trades}/{len(actual_prices)} profitable trades "
            f"({accuracy_score:.2%}), total profit: ${total_profit:.2f}"
        )
        
    else:
        # === GENERAL MODE (NO TRADING METRICS) ===
        logger.debug("General evaluation mode - skipping trading metrics")
        total_buy_profit = total_sell_profit = total_profit = profit_per_trade = 0
        profitable_trades = 0
        accuracy_score = 0
        buy_profits = sell_profits = []
    
    # === CALCULATE MEAN ABSOLUTE ERROR ===
    if scale:
        # Data is scaled - calculate MAE directly (assumes data is inverse-transformed)
        mae = np.mean(np.abs(actual_prices - predicted_prices))
        logger.debug(f"Calculated MAE from scaled data: {mae:.4f}")
    else:
        # Data is not scaled - prefer model's loss value if available
        if loss_value is not None:
            mae = loss_value
            logger.debug(f"Using model loss as MAE: {mae:.4f}")
        else:
            # Fallback to direct calculation
            mae = np.mean(np.abs(actual_prices - predicted_prices))
            logger.debug(f"Calculated MAE directly: {mae:.4f}")
    
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


def _aggregate_multi_feature_metrics(
    all_metrics: Dict[str, Dict[str, Any]], 
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Aggregate evaluation metrics across multiple features.
    
    When evaluating models that predict multiple features (e.g., OHLCV data),
    this function combines the individual feature metrics into overall performance metrics.
    
    Args:
        all_metrics: Dictionary mapping feature names to their individual metrics
        feature_names: List of all feature names being evaluated
    
    Returns:
        Dictionary containing:
        - mean_absolute_error: Average MAE across all features
        - total_profit: Combined profit from all price-related features
        - profit_per_trade: Average profit per trade across price features
        - feature_metrics: Individual metrics for each feature
        - feature_names: List of feature names for reference
    """
    
    # === AGGREGATE BASIC METRICS ===
    # Calculate average MAE across all features
    total_mae = sum(metrics['mean_absolute_error'] for metrics in all_metrics.values())
    avg_mae = total_mae / len(feature_names)
    
    logger.debug(f"Average MAE across {len(feature_names)} features: {avg_mae:.4f}")
    
    # === AGGREGATE TRADING METRICS ===
    # Only aggregate trading metrics for price-related features
    price_keywords = ["price", "close", "open", "high", "low"]
    price_features = [
        name for name in feature_names 
        if any(keyword in name.lower() for keyword in price_keywords)
    ]
    
    if price_features:
        logger.debug(f"Aggregating trading metrics for price features: {price_features}")
        
        # Sum profits and trades across all price features
        total_profit = sum(all_metrics[name]['total_profit'] for name in price_features)
        total_trades = sum(all_metrics[name]['total_trades'] for name in price_features)
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        
        logger.debug(
            f"Combined trading results: ${total_profit:.2f} total profit "
            f"across {total_trades} trades (${avg_profit_per_trade:.4f} per trade)"
        )
    else:
        logger.debug("No price features found - trading metrics set to zero")
        total_profit = avg_profit_per_trade = 0
    
    return {
        'mean_absolute_error': avg_mae,
        'total_profit': total_profit,
        'profit_per_trade': avg_profit_per_trade,
        'feature_metrics': all_metrics,
        'feature_names': feature_names
    }


def _print_and_save_results(
    metrics: Dict[str, Any],
    future_price: float,
    loss_value: float,
    loss_name: str,
    lookup_step: int,
    filename: Optional[str]
) -> None:
    """
    Print evaluation results to console and optionally save to file.
    
    Args:
        metrics: Dictionary containing calculated metrics
        future_price: Predicted future price for display
        loss_value: Model loss value
        loss_name: Name of the loss function
        lookup_step: Prediction horizon in days
        filename: Optional file path to save results
    """
    
    # === CONSOLE OUTPUT ===
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Basic prediction info
    day_text = "day" if lookup_step == 1 else f"{lookup_step} days"
    print(f"Future price after {day_text}: ${future_price:.2f}")
    print(f"{loss_name}: {loss_value:.4f}")
    print(f"Mean Absolute Error: {metrics['mean_absolute_error']:.4f}")
    
    # Trading metrics (if available)
    if 'accuracy_score' in metrics:
        print("\nTRADING PERFORMANCE:")
        print(f"  Trading Accuracy: {metrics['accuracy_score']:.2%}")
        print(f"  Total Buy Profit: ${metrics['total_buy_profit']:.2f}")
        print(f"  Total Sell Profit: ${metrics['total_sell_profit']:.2f}")
        print(f"  Total Profit: ${metrics['total_profit']:.2f}")
        print(f"  Profit per Trade: ${metrics['profit_per_trade']:.4f}")
        print(f"  Profitable Trades: {metrics.get('profitable_trades', 'N/A')}/{metrics.get('total_trades', 'N/A')}")
    
    # Multi-feature breakdown (if available)
    if 'feature_metrics' in metrics:
        print(f"\nMULTI-FEATURE BREAKDOWN:")
        for feature_name, feature_metrics in metrics['feature_metrics'].items():
            print(f"  {feature_name}:")
            print(f"    MAE: {feature_metrics['mean_absolute_error']:.4f}")
            if 'accuracy_score' in feature_metrics:
                print(f"    Trading Accuracy: {feature_metrics['accuracy_score']:.2%}")
                print(f"    Total Profit: ${feature_metrics['total_profit']:.2f}")
    
    print("="*50)
    
    # === FILE OUTPUT ===
    if filename:
        try:
            with open(filename, 'w') as f:
                f.write("EVALUATION RESULTS\n")
                f.write("="*50 + "\n\n")
                
                # Basic metrics
                f.write(f"Future price after {day_text}: ${future_price:.2f}\n")
                f.write(f"{loss_name}: {loss_value:.4f}\n")
                f.write(f"Mean Absolute Error: {metrics['mean_absolute_error']:.4f}\n\n")
                
                # Trading metrics
                if 'accuracy_score' in metrics:
                    f.write("TRADING PERFORMANCE:\n")
                    f.write(f"Trading Accuracy: {metrics['accuracy_score']:.2%}\n")
                    f.write(f"Total Buy Profit: ${metrics['total_buy_profit']:.2f}\n")
                    f.write(f"Total Sell Profit: ${metrics['total_sell_profit']:.2f}\n")
                    f.write(f"Total Profit: ${metrics['total_profit']:.2f}\n")
                    f.write(f"Profit per Trade: ${metrics['profit_per_trade']:.4f}\n")
                    f.write(f"Profitable Trades: {metrics.get('profitable_trades', 'N/A')}/{metrics.get('total_trades', 'N/A')}\n\n")
                
                # Multi-feature breakdown
                if 'feature_metrics' in metrics:
                    f.write("MULTI-FEATURE BREAKDOWN:\n")
                    for feature_name, feature_metrics in metrics['feature_metrics'].items():
                        f.write(f"{feature_name}:\n")
                        f.write(f"  MAE: {feature_metrics['mean_absolute_error']:.4f}\n")
                        if 'accuracy_score' in feature_metrics:
                            f.write(f"  Trading Accuracy: {feature_metrics['accuracy_score']:.2%}\n")
                            f.write(f"  Total Profit: ${feature_metrics['total_profit']:.2f}\n")
                        f.write("\n")
            
            logger.info(f"Results saved to {filename}")
            print(f"Results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results to {filename}: {e}")
            print(f"Warning: Could not save results to file: {e}")

