"""
Simple Trading Evaluation Utilities

This module provides a simple function to calculate trading metrics for stock prediction models.
"""

import numpy as np


def calculate_trading_metrics(actual_prices, predicted_prices, current_prices=None, 
                            lookup_step=1, future_price=None, loss_value=None, 
                            loss_name="loss", filename=None, scale=True, 
                            feature_names=None, evaluation_mode="price"):
    """
    Calculate trading accuracy and profit metrics.
    
    Simple trading strategy:
    - BUY when predicted > current (profit = actual - current)
    - SELL when predicted < current (profit = current - actual)
    """
    actual_prices = np.array(actual_prices).flatten()
    predicted_prices = np.array(predicted_prices).flatten()
    
    if current_prices is None:
        current_prices = actual_prices
    else:
        current_prices = np.array(current_prices).flatten()
    
    # Calculate buy and sell profits
    buy_profits = []
    sell_profits = []
    
    for i in range(len(actual_prices)):
        current = current_prices[i]
        pred_future = predicted_prices[i]
        true_future = actual_prices[i]
        
        # Buy if predicted price increase
        if pred_future > current:
            buy_profit = true_future - current
        else:
            buy_profit = 0
            
        # Sell if predicted price decrease  
        if pred_future < current:
            sell_profit = current - true_future
        else:
            sell_profit = 0
            
        buy_profits.append(buy_profit)
        sell_profits.append(sell_profit)
    
    # Calculate metrics
    total_buy_profit = sum(buy_profits)
    total_sell_profit = sum(sell_profits)
    total_profit = total_buy_profit + total_sell_profit
    profit_per_trade = total_profit / len(actual_prices)
    
    # Count profitable trades
    profitable_trades = sum(1 for bp, sp in zip(buy_profits, sell_profits) if bp > 0 or sp > 0)
    accuracy_score = profitable_trades / len(actual_prices)
    
    # Mean absolute error
    mean_absolute_error = np.mean(np.abs(actual_prices - predicted_prices))
    
    results = {
        'accuracy_score': accuracy_score,
        'total_buy_profit': total_buy_profit,
        'total_sell_profit': total_sell_profit,
        'total_profit': total_profit,
        'profit_per_trade': profit_per_trade,
        'mean_absolute_error': mean_absolute_error,
        'profitable_trades': profitable_trades,
        'total_trades': len(actual_prices),
        'buy_profits': buy_profits,
        'sell_profits': sell_profits
    }
    
    # Print results if requested
    if future_price is not None and loss_value is not None:
        day_text = "day" if lookup_step == 1 else f"{lookup_step} days"
        print(f"Future price after {day_text}: ${future_price:.2f}")
        print(f"{loss_name}: {loss_value:.4f}")
        print(f"Mean Absolute Error: {mean_absolute_error:.4f}")
        print(f"Accuracy score: {accuracy_score:.4f}")
        print(f"Total buy profit: ${total_buy_profit:.2f}")
        print(f"Total sell profit: ${total_sell_profit:.2f}")
        print(f"Total profit: ${total_profit:.2f}")
        print(f"Profit per trade: ${profit_per_trade:.4f}")
    
    # Save to file if requested
    if filename:
        try:
            with open(filename, 'w') as f:
                f.write(f"Future price after {day_text}: ${future_price:.2f}\n")
                f.write(f"{loss_name}: {loss_value:.4f}\n")
                f.write(f"Mean Absolute Error: {mean_absolute_error:.4f}\n")
                f.write(f"Accuracy score: {accuracy_score:.4f}\n")
                f.write(f"Total buy profit: ${total_buy_profit:.2f}\n")
                f.write(f"Total sell profit: ${total_sell_profit:.2f}\n")
                f.write(f"Total profit: ${total_profit:.2f}\n")
                f.write(f"Profit per trade: ${profit_per_trade:.4f}\n")
        except:
            pass
    
    return results