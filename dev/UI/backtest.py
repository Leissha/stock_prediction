import pandas as pd
import numpy as np

def backtest(predictions, actual_prices):
    portfolio = [100]  # Starting portfolio value
    for i in range(1, len(predictions)):
        if predictions[i] > actual_prices[i - 1]:  # Buy signal
            portfolio.append(portfolio[-1] * (1 + (actual_prices[i] - actual_prices[i - 1]) / actual_prices[i - 1]))
        else:  # Hold
            portfolio.append(portfolio[-1])
    return portfolio
