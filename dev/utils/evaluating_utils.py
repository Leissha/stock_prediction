"""
Trading Evaluation Utilities to calculate trading metrics for stock prediction models.
"""

import numpy as np

def trading_metrics_from_prices(p_true, p_pred, threshold=0.01, cost=0.0005):
    """
    Clean trading metrics from price sequences (Rule B).
    
    Args:
        p_true: True price sequence (inverse-transformed)
        p_pred: Predicted price sequence (inverse-transformed) 
        threshold: Return threshold to trigger trade
        cost: Transaction cost per trade (as fraction)
    
    Returns:
        Dictionary with trading metrics in percent space
    """
    # Convert to numpy arrays
    p_true = np.array(p_true).flatten()
    p_pred = np.array(p_pred).flatten()
    
    if len(p_true) < 2:
        return {'accuracy': 0.0, 'total_profit_pct': 0.0, 'profit_per_trade_pct': 0.0, 
                'profitable_trades': 0, 'total_trades': 1}
    
    # 1-step returns on true price
    r_true = (p_true[1:] - p_true[:-1]) / p_true[:-1]
    # Model-implied next-step returns (use current price in denominator)
    r_pred = (p_pred[1:] - p_true[:-1]) / p_true[:-1]
    
    # Anti-leakage check
    assert not np.allclose(p_pred[1:], p_true[1:]), "Data leakage detected: predictions == targets"
    
    # Correlation diagnostic
    corr = np.corrcoef(r_pred, r_true)[0,1] if len(r_pred) > 1 else 0.0
    
    # Signals based on predicted returns
    sig = np.where(r_pred > threshold, 1, 
                   np.where(r_pred < -threshold, -1, 0))
    trade_mask = sig != 0
    
    if not trade_mask.any():
        return {'accuracy': 0.0, 'total_profit_pct': 0.0, 'profit_per_trade_pct': 0.0, 
                'profitable_trades': 0, 'total_trades': 1, 'correlation': corr}
    
    # Per-trade P&L in percent; apply cost per executed trade
    pnl = sig[trade_mask] * r_true[trade_mask] - cost
    
    # Metrics
    total_profit_pct = float(pnl.sum())  # % return if you invest 1 unit per trade
    profit_per_trade = float(pnl.mean()) if pnl.size else 0.0
    accuracy = float((np.sign(sig[trade_mask]) == np.sign(r_true[trade_mask])).mean()) if pnl.size else 0.0
    profitable_trades = int((pnl > 0).sum())
    total_trades = int(trade_mask.sum())
    
    return {
        'accuracy': accuracy,
        'total_profit_pct': total_profit_pct, 
        'profit_per_trade_pct': profit_per_trade,
        'profitable_trades': profitable_trades,
        'total_trades': total_trades,
        'correlation': corr,
        'signals': sig,
        'pnl_pct': pnl
    }

# Backward compatibility wrapper 
def calculate_trading_metrics_returns(actual_returns, predicted_returns, current_prices, threshold=0.002, cost=0.001):
    """Legacy wrapper - converts to price-based evaluation."""
    # This should not be used anymore - keeping for compatibility
    import warnings
    warnings.warn("Use trading_metrics_from_prices instead", DeprecationWarning)
    
    # Convert returns back to prices (approximate)
    current_prices = np.array(current_prices).flatten()
    actual_returns = np.array(actual_returns).flatten() 
    predicted_returns = np.array(predicted_returns).flatten()
    
    if len(current_prices) == 0:
        return {'accuracy_score': 0.0, 'total_profit': 0.0}
        
    # Reconstruct price sequences
    p_true = current_prices * (1 + actual_returns)
    p_pred = current_prices * (1 + predicted_returns)
    
    # Use new clean function
    result = trading_metrics_from_prices(
        np.concatenate([current_prices[:1], p_true]), 
        np.concatenate([current_prices[:1], p_pred]),
        threshold=threshold, cost=cost
    )
    
    # Map to old field names for compatibility
    return {
        'accuracy_score': result['accuracy'],
        'total_profit': result['total_profit_pct'] * np.mean(current_prices),  # Approximate dollars
        'profit_per_trade': result['profit_per_trade_pct'],
        'profitable_trades': result['profitable_trades'],
        'total_trades': result['total_trades'],
        'total_samples': len(actual_returns)
    }