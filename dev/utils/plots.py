import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_predictions(actual_prices, predicted_prices, ticker, save_path="results/price_chart.png", dates=None):
    """
    Plot the actual and predicted prices
    Args:
        actual_prices: The actual prices
        predicted_prices: The predicted prices
        ticker: The ticker of the stock
        dates: The dates of the prices
        save_path: The path to save the plot
    """
    if dates is None:
        dates = range(len(actual_prices))
    plt.plot(dates, actual_prices, color="black", label=f"Actual {ticker} Price")
    plt.plot(dates, predicted_prices, color="green", label=f"Predicted {ticker} Price")
    plt.xlabel("Date")
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.title(f"{ticker} Share Price")
    plt.ylabel(f"{ticker} Share Price")
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()  # Display the plot
    plt.close()
    
# def create_candlestick_chart(df, ticker, save_path="inspect_data/candlestick_chart.png", n_days=1):
#     """
#     Create candlestick chart for stock data.
#     Each candlestick can represent n trading days (aggregated OHLC).
#     """
#     # Ensure datetime index for resampling
#     df.index = pd.to_datetime(df.index)

#     # Resample if n_days > 1 (aggregate OHLC values)
#     if n_days > 1:
#         df = df.resample(f'{n_days}D').agg({
#             'open': 'first',
#             'close': 'last',
#             'high': 'max',
#             'low': 'min'
#         }).dropna()

#     candlestick = go.Candlestick(
#         x=df.index,
#         open=df['open'],
#         high=df['high'],
#         low=df['low'],
#         close=df['close'],
#         name="Candlestick"
#     )

#     fig = go.Figure(data=[candlestick])

#     fig.update_layout(
#         width=800, height=600,
#         title=f"{ticker} Candlestick Chart ({n_days}-day)",
#         yaxis_title=f'{ticker} Stock Price',
#     )

#     # Ensure save directory exists
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     fig.write_image(save_path)   # needs kaleido installed
#     fig.show()

def create_candlestick_chart(df, ticker, save_path="inspect_data/candlestick_chart.png", n_days=1):
    """
    Create candlestick chart with SMA and EMA.
    Each candlestick can represent n trading days (aggregated OHLC).
    """
    # Ensure datetime index
    df.index = pd.to_datetime(df.index)

    # Aggregate OHLC values if n_days > 1  
    if n_days > 1:
        df = df.resample(f'{n_days}D').agg({
            'open': 'first', # first value in the window
            'close': 'last', # last value in the window
            'high': 'max',   # max value in the window
            'low': 'min'     # min value in the window
        }).dropna()

    # Add SMA and EMA if they don't exist
    if 'SMA' not in df.columns:
        df['SMA'] = df['close'].rolling(window=20, min_periods=1).mean()  # min_periods=1 to handle NaN
    if 'EMA' not in df.columns:
        df['EMA'] = df['close'].ewm(span=20, min_periods=1).mean()  # min_periods=1 to handle NaN

    # Base candlestick
    candlestick = go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candlestick"
    )

    # Simple Moving Average (SMA)
    sma = go.Scatter(
        x=df.index,
        y=df['SMA'],
        mode='lines',
        line=dict(color='blue'),
        name="SMA (20)"
    )

    # Exponential Moving Average (EMA)
    ema = go.Scatter(
        x=df.index,
        y=df['EMA'],
        mode='lines',
        line=dict(color='orange'),
        name="EMA (20)"
    )

    # Build figure
    fig = go.Figure(data=[candlestick, sma, ema])
    fig.update_layout(
        width=900, height=600,
        title=f"{ticker} Candlestick with SMA & EMA",
        yaxis_title=f'{ticker} Stock Price',
        xaxis_title="Date",
        xaxis_rangeslider_visible=False  
    )

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig.write_image(save_path)  
    fig.show()
    
def create_boxplot(df, ticker, save_path="inspect_data/boxplot.png", n_days=20):
    """
    Create boxplots of stock closing prices for consecutive n-day windows.
    Shows distribution of prices (median, quartiles, outliers).
    """
    traces = []
    df.index = pd.to_datetime(df.index)

    for i in range(0, len(df) - n_days + 1, n_days):
        window_data = df.iloc[i:i+n_days]['close'].dropna()
        period_name = f"{df.index[i].strftime('%d-%m-%y')} → {df.index[i+n_days-1].strftime('%d-%m-%y')}"

        traces.append(go.Box(
            y=window_data,
            name=period_name,
            boxpoints='outliers'
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        width=900, height=600,
        title=f"{ticker} Closing Price Distribution ({n_days}-Trading-Day windows)",
        yaxis_title=f'{ticker} Stock Price',
        xaxis_title="Time Windows",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            tickangle=45
        )
    )

    fig.write_image(save_path)   
    fig.show()