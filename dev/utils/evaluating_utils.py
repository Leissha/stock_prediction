def calculate_trading_metrics(
    actual_prices, 
    predicted_prices, 
    current_prices=None,
    lookup_step=1, 
    future_price=None, 
    loss_val=None, 
    mae_val=None,
    rmse_val=None,
    filename=None,
    scale=True,
    training_features=None,
    target_feature=None,
):
    """
    Improved, simple trading metrics:
    - One action per step (buy OR sell), else no-trade.
    - No-trade zone via threshold on predicted return.
    - Flat transaction cost per round-trip.
    - Adds directional_accuracy on executed trades.
    """
    import numpy as np  # ensure available even if file import is edited

    actual_prices = np.array(actual_prices).flatten()
    predicted_prices = np.array(predicted_prices).flatten()

    # --- hyperparams (simple & editable) ---
    threshold = 0.0005    # 0.05% no-trade zone (reduced for return predictions)
    trade_cost = 0.001    # 0.1% per side → ~0.2% round-trip

    buy_profits, sell_profits = [], []
    num_trades = 0
    direction_hits = 0
    profitable_trades_cnt = 0  # for backward-compatible accuracy_score

    for i in range(len(actual_prices)):
        # current reference price
        if current_prices is not None and i < len(current_prices):
            current = float(current_prices[i])
        else:
            current = float(actual_prices[i-1] if i > 0 else actual_prices[i])

        pred_future = float(predicted_prices[i])
        true_future = float(actual_prices[i])

        # predicted / true returns vs current
        pred_ret = (pred_future - current) / (current if current != 0 else 1.0)
        true_ret = (true_future - current) / (current if current != 0 else 1.0)

        if pred_ret > threshold:
            # BUY 1 unit, subtract round-trip cost ~2 * trade_cost * current
            pnl = (true_future - current) - (current * 2 * trade_cost)
            buy_profits.append(max(pnl, 0.0))
            sell_profits.append(0.0)
            num_trades += 1
            direction_hits += int(true_ret > 0)
            profitable_trades_cnt += int(pnl > 0)
        elif pred_ret < -threshold:
            # SELL 1 unit
            pnl = (current - true_future) - (current * 2 * trade_cost)
            sell_profits.append(max(pnl, 0.0))
            buy_profits.append(0.0)
            num_trades += 1
            direction_hits += int(true_ret < 0)
            profitable_trades_cnt += int(pnl > 0)
        else:
            # no trade
            buy_profits.append(0.0)
            sell_profits.append(0.0)

    total_buy_profit = float(np.sum(buy_profits))
    total_sell_profit = float(np.sum(sell_profits))
    total_profit = total_buy_profit + total_sell_profit

    if num_trades > 0:
        profit_per_trade = total_profit / num_trades
        directional_accuracy = direction_hits / num_trades
        # Backward-compat “accuracy_score”: profitable_trades / total_trades_executed
        accuracy_score = profitable_trades_cnt / num_trades
    else:
        profit_per_trade = 0.0
        directional_accuracy = 0.0
        accuracy_score = 0.0

    results = {
        # new but handy
        'directional_accuracy': directional_accuracy,
        # legacy/compatible keys
        'accuracy_score': accuracy_score,
        'total_buy_profit': total_buy_profit,
        'total_sell_profit': total_sell_profit,
        'total_profit': total_profit,
        'profit_per_trade': profit_per_trade,
        'profitable_trades': profitable_trades_cnt,
        'total_trades': num_trades,
        'buy_profits': buy_profits,
        'sell_profits': sell_profits
    }

    if filename:
        try:
            with open(filename, 'w') as f:
                f.write("TEST DATASET PERFORMANCE METRICS\n")
                f.write("===============================\n")
                if future_price is not None:
                    f.write(f"Latest predicted price: ${float(future_price):.2f}\n")
                if loss_val is not None:
                    f.write(f"Model loss on test data: {loss_val} (MSE)\n")
                if mae_val is not None:
                    f.write(f"Mean Absolute Error: {mae_val}\n")
                if rmse_val is not None:
                    f.write(f"Root Mean Squared Error: {rmse_val}\n")
                f.write(f"Directional accuracy (on executed trades): {directional_accuracy:.4f}\n")
                f.write(f"Trading accuracy (profitable trades): {accuracy_score:.4f}\n")
                f.write(f"Total buy profit: ${total_buy_profit:.2f}\n")
                f.write(f"Total sell profit: ${total_sell_profit:.2f}\n")
                f.write(f"Total profit: ${total_profit:.2f}\n")
                f.write(f"Profit per trade: ${profit_per_trade:.4f}\n")
                f.write(f"Profitable trades: {profitable_trades_cnt}/{num_trades} executed trades\n")
                f.write(f"Trading threshold: {threshold*100:.2f}% | Round-trip cost: ~{2*trade_cost*100:.2f}%\n")
                f.write("\nMODEL CONFIGURATION\n")
                f.write("==================\n")
                f.write(f"Training features: {training_features}\n")
                f.write(f"Target feature: {target_feature}\n")
                f.write(f"Feature scaling applied: {scale}\n")
        except Exception as e:
            print(f"Error saving trading metrics: {e}")

    return results
