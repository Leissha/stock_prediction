"""
Trading Evaluation Utilities to calculate trading metrics for stock prediction models.
"""

import numpy as np

def calculate_trading_metrics(
    actual_prices, 
    predicted_prices, 
    current_prices=None,
    lookup_step=1, 
    future_price=None, 
    loss_val=None, 
    filename=None,
    scale=True,
    training_features=None,
    target_feature=None,
):
    """
    Calculate trading accuracy and profit metrics.
    
    Simple trading strategy:
    - BUY when predicted > current (profit = actual - current)
    - SELL when predicted < current (profit = current - actual)
    
    Args:
        actual_prices: True values from test data
        predicted_prices: Model predictions
        current_prices: Day before each prediction
        lookup_step: Prediction horizon in days
        future_price: Single latest prediction value.
        loss_value: Model's loss metric for reporting
        filename: Path to save results as text file
    
    Returns:
        Dictionary containing trading metrics
    """
    actual_prices = np.array(actual_prices).flatten()
    predicted_prices = np.array(predicted_prices).flatten()
    
    # Define lambda functions for profit calculations with minimum threshold
    # Only trade if prediction differs by more than 0.5% from current price
    threshold = 0.005  # 0.5% threshold
    
    buy_profit_calc = lambda current, pred_future, true_future: true_future - current if pred_future > current * (1 + threshold) else 0
    sell_profit_calc = lambda current, pred_future, true_future: current - true_future if pred_future < current * (1 - threshold) else 0
    
    # Calculate buy and sell profits using lambda functions
    buy_profits = []
    sell_profits = []
    
    for i in range(len(actual_prices)):
        # Use provided current prices or fallback to previous day's price
        if current_prices is not None and i < len(current_prices):
            current = current_prices[i]
        else:
            # Fallback: use previous day's actual price as current
            current = actual_prices[i-1] if i > 0 else actual_prices[i]
        pred_future = predicted_prices[i]
        true_future = actual_prices[i]
        
        buy_profits.append(buy_profit_calc(current, pred_future, true_future))
        sell_profits.append(sell_profit_calc(current, pred_future, true_future))
    
    # Calculate metrics
    total_buy_profit = sum(buy_profits)
    total_sell_profit = sum(sell_profits)
    total_profit = total_buy_profit + total_sell_profit
    profit_per_trade = total_profit / len(actual_prices)
    
    # Count profitable trades
    profitable_trades = sum(1 for bp, sp in zip(buy_profits, sell_profits) if bp > 0 or sp > 0)
    accuracy_score = profitable_trades / len(actual_prices)
    
    results = {
        'accuracy_score': accuracy_score,
        'total_buy_profit': total_buy_profit,
        'total_sell_profit': total_sell_profit,
        'total_profit': total_profit,
        'profit_per_trade': profit_per_trade,
        'profitable_trades': profitable_trades,
        'total_trades': len(actual_prices),
        'buy_profits': buy_profits,
        'sell_profits': sell_profits
    }

    # Save to file if requested
    if filename:
        try:
            with open(filename, 'w') as f:
                f.write(f"TEST DATASET PERFORMANCE METRICS\n")
                f.write(f"===============================\n")
                f.write(f"Latest predicted price: ${future_price if future_price is not None else 0:.2f}\n")
                f.write(f"Model loss on test data: {loss_val if loss_val is not None else 'N/A'}\n")
                f.write(f"Trading accuracy (profitable trades): {accuracy_score:.4f}\n")
                f.write(f"Total buy profit: ${total_buy_profit:.2f}\n")
                f.write(f"Total sell profit: ${total_sell_profit:.2f}\n")
                f.write(f"Total profit: ${total_profit:.2f}\n")
                f.write(f"Profit per trade: ${profit_per_trade:.4f}\n")
                f.write(f"Profitable trades: {profitable_trades}/{len(actual_prices)} test samples\n")
                f.write(f"Trading threshold: 0.5%\n")
                f.write(f"\nMODEL CONFIGURATION\n")
                f.write(f"==================\n")
                f.write(f"Training features: {training_features if training_features is not None else 'N/A'}\n")
                f.write(f"Target feature: {target_feature}\n")
                f.write(f"Feature scaling applied: {scale}\n")
            print(f"Trading metrics saved to: {filename}")
        except Exception as e:
            print(f"Error saving trading metrics: {e}")

    return results