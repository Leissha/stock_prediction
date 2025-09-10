import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

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
    
# Simple candlestick chart
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
    Create candlestick chart with SMA and EMA computed on a COPY of df.
    This function is PURE: it must not mutate the caller's dataframe.
    """
    # Work on a local copy only
    loc = df[["open", "high", "low", "close"]].copy()
    loc.index = pd.to_datetime(loc.index, errors='coerce', utc=False)

    # Aggregate OHLC values if n_days > 1
    if n_days > 1:
        loc = loc.resample(f'{n_days}D').agg({
            'open': 'first',
            'close': 'last',
            'high': 'max',
            'low': 'min'
        }).dropna()

    # Compute SMA and EMA for plotting only (do not add to original df)
    sma_series = loc['close'].rolling(window=20, min_periods=1).mean()
    ema_series = loc['close'].ewm(span=20, min_periods=1).mean()

    # Convert to plain Python lists (PNG export can't handle pandas objects)
    x_list = loc.index.to_pydatetime().tolist()
    open_list = loc["open"].tolist()
    high_list = loc["high"].tolist()
    low_list = loc["low"].tolist()
    close_list = loc["close"].tolist()
    sma_list = sma_series.tolist()
    ema_list = ema_series.tolist()

    # Base candlestick
    candlestick = go.Candlestick(
        x=x_list,
        open=open_list,
        high=high_list,
        low=low_list,
        close=close_list,
        name="Candlestick"
    )

    # Simple Moving Average (SMA)
    sma = go.Scatter(
        x=x_list,
        y=sma_list,
        mode='lines',
        line=dict(color='blue'),
        name="SMA (20)"
    )

    # Exponential Moving Average (EMA)
    ema = go.Scatter(
        x=x_list,
        y=ema_list,
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
    fig.update_xaxes(type="date", tickformat="%Y-%m-%d")

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig.write_image(save_path)  
    # web browser html options
    # pio.renderers.default = "browser"
    # fig.show()
    
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
    # web browser html options
    # pio.renderers.default = "browser"
    # fig.show()