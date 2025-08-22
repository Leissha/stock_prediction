import pandas as pd
import numpy as np

def calculate_trading_accuracy(actual_prices, predicted_prices, current_prices=None, lookup_step=1):
    """
    Calculate trading-based accuracy metrics using the same logic as p1.
    
    Args:
        actual_prices: Array of actual future prices
        predicted_prices: Array of predicted future prices  
        current_prices: Array of current prices (if None, will use actual_prices as baseline)
        lookup_step: Number of days into the future being predicted (default=1 for next day)
    
    Returns:
        dict: Dictionary containing all accuracy metrics
    """
    # If current_prices not provided, use actual_prices as baseline (for v0.1 case)
    if current_prices is None:
        current_prices = actual_prices
    
    # Ensure all arrays are 1D and have same length
    actual_prices = np.array(actual_prices).flatten()
    predicted_prices = np.array(predicted_prices).flatten()
    current_prices = np.array(current_prices).flatten()
    
    if len(actual_prices) != len(predicted_prices) or len(actual_prices) != len(current_prices):
        raise ValueError("All price arrays must have the same length")
    
    # Define profit calculation functions (same as p1)
    buy_profit = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    
    # Calculate profits for each prediction
    buy_profits = []
    sell_profits = []
    
    for i in range(len(actual_prices)):
        current = current_prices[i]
        pred_future = predicted_prices[i]
        true_future = actual_prices[i]
        
        buy_profits.append(buy_profit(current, pred_future, true_future))
        sell_profits.append(sell_profit(current, pred_future, true_future))
    
    # Calculate metrics
    total_buy_profit = sum(buy_profits)
    total_sell_profit = sum(sell_profits)
    total_profit = total_buy_profit + total_sell_profit
    profit_per_trade = total_profit / len(actual_prices) if len(actual_prices) > 0 else 0
    
    # Calculate accuracy score (percentage of profitable trades)
    profitable_trades = sum(1 for bp, sp in zip(buy_profits, sell_profits) if bp > 0 or sp > 0)
    accuracy_score = profitable_trades / len(actual_prices) if len(actual_prices) > 0 else 0
    
    # Calculate Mean Absolute Error
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

def print_accuracy_metrics(metrics, future_price, loss_value, loss_name="loss", lookup_step=1):
    """
    Print accuracy metrics in the same format as p1.
    
    Args:
        metrics: Dictionary returned by calculate_trading_accuracy
        future_price: Predicted future price
        loss_value: Model loss value
        loss_name: Name of the loss function (default="loss")
        lookup_step: Number of days into the future (default=1)
    """
    print(f"Future price after {lookup_step} day{'s' if lookup_step > 1 else ''} is {future_price:.2f}$")
    print(f"{loss_name}: {loss_value}")
    print(f"Mean Absolute Error: {metrics['mean_absolute_error']}")
    print(f"Accuracy score: {metrics['accuracy_score']}")
    print(f"Total buy profit: {metrics['total_buy_profit']}")
    print(f"Total sell profit: {metrics['total_sell_profit']}")
    print(f"Total profit: {metrics['total_profit']}")
    print(f"Profit per trade: {metrics['profit_per_trade']}")

def save_accuracy_to_csv(metrics, future_price, loss_value, loss_name, filename, lookup_step=1):
    """
    Save accuracy metrics to CSV file in the same format as p1.
    
    Args:
        metrics: Dictionary returned by calculate_trading_accuracy
        future_price: Predicted future price
        loss_value: Model loss value
        loss_name: Name of the loss function
        filename: Output CSV filename
        lookup_step: Number of days into the future (default=1)
    """
    with open(filename, 'w') as f:
        f.write(f"Future price after {lookup_step} day{'s' if lookup_step > 1 else ''} is {future_price:.2f}$\n")
        f.write(f"{loss_name}: {loss_value}\n")
        f.write(f"Mean Absolute Error: {metrics['mean_absolute_error']}\n")
        f.write(f"Accuracy score: {metrics['accuracy_score']}\n")
        f.write(f"Total buy profit: {metrics['total_buy_profit']}\n")
        f.write(f"Total sell profit: {metrics['total_sell_profit']}\n")
        f.write(f"Total profit: {metrics['total_profit']}\n")
        f.write(f"Profit per trade: {metrics['profit_per_trade']}\n")
