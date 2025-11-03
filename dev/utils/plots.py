import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def plot_predictions(actual_prices, predicted_prices, ticker, save_path, dates=None):
    """
    Plot the actual and predicted prices
    Args:
        actual_prices: The actual prices
        predicted_prices: The predicted prices
        ticker: The ticker of the stock
        dates: The dates of the prices
        save_path: The path to save the plot
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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
    plt.close()

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

def plot_training_metrics(history, save_path="results/training_metrics.png"):
    """
    Plot training metrics (Loss, MAE, RMSE if available)
    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    # Validate history object
    if history is None:
        print("Warning: history is None, skipping training metrics plot")
        return
    
    if not hasattr(history, 'history') or not history.history:
        print("Warning: history.history is empty, skipping training metrics plot")
        return
    
    if 'loss' not in history.history or not history.history['loss']:
        print("Warning: No loss data in history, skipping training metrics plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Plot multiple metrics (train + validation if available)
    plt.plot(epochs, history.history['loss'], color='blue', linewidth=2, label='Train Loss')
    if 'val_loss' in history.history and history.history['val_loss']:
        plt.plot(epochs, history.history['val_loss'], color='red', linewidth=2, label='Val Loss')
    
    plt.title('Training Metrics Over Epochs')
    plt.xlabel('Epoch')
    # Choose a sensible y-label based on available metrics
    if 'accuracy' in history.history:
        ylabel = 'Binary Crossentropy (loss)'
    elif 'mae' in history.history:
        ylabel = 'MSE Loss'
    else:
        ylabel = 'Loss'
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training metrics plot saved to: {save_path}")


def plot_sentiment(bundle, save_path, sentiment_span: int = 15):
    """
    Single plot with train/val/test price and smoothed sentiment distinguished by color.
    Requires merged sentiment columns (e.g., 'sentiment_mean') in bundle DataFrames.
    """
    parts = []
    if hasattr(bundle, 'train_df') and bundle.train_df is not None and 'sentiment_mean' in bundle.train_df.columns:
        df = bundle.train_df.copy()
        df['__split__'] = 'train'
        parts.append(df)
    if hasattr(bundle, 'val_df') and bundle.val_df is not None and 'sentiment_mean' in bundle.val_df.columns:
        df = bundle.val_df.copy()
        df['__split__'] = 'val'
        parts.append(df)
    if hasattr(bundle, 'test_df') and bundle.test_df is not None and 'sentiment_mean' in bundle.test_df.columns:
        df = bundle.test_df.copy()
        df['__split__'] = 'test'
        parts.append(df)
    if not parts:
        return

    df_all = pd.concat(parts, axis=0)
    try:
        df_all.index = pd.to_datetime(df_all.index)
    except Exception:
        pass
    df_all = df_all.sort_index()

    # Smooth per split
    df_all['__sent_smooth__'] = (
        df_all.groupby('__split__')['sentiment_mean']
        .apply(lambda s: s.ewm(span=max(1, sentiment_span)).mean())
        .reset_index(level=0, drop=True)
    )

    fig, ax1 = plt.subplots(figsize=(12, 5))
    color_map = {'train': 'tab:green', 'val': 'tab:orange', 'test': 'tab:blue'}

    # Price lines
    for split, grp in df_all.groupby('__split__'):
        x = grp.index.to_pydatetime().tolist() if isinstance(grp.index, pd.DatetimeIndex) else list(grp.index)
        # Prefer explicit price columns; never fall back to an arbitrary first column
        if 'close' in grp.columns:
            y_series = grp['close']
        elif 'adj close' in grp.columns:
            y_series = grp['adj close']
        elif 'adj_close' in grp.columns:
            y_series = grp['adj_close']
        else:
            # If no recognized price column, skip this split to avoid plotting returns by mistake
            continue
        y = y_series.astype(float).tolist()
        # Ensure split is a str for dict.get typing
        split_key = str(split)
        ax1.plot(x, y, color=color_map.get(split_key, 'gray'), linewidth=2, label=f'Price ({split_key})')

    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.25)

    # Sentiment dashed
    ax2 = ax1.twinx()
    for split, grp in df_all.groupby('__split__'):
        x = grp.index.to_pydatetime().tolist() if isinstance(grp.index, pd.DatetimeIndex) else list(grp.index)
        y = grp['__sent_smooth__'].astype(float).tolist()
        split_key = str(split)
        ax2.plot(x, y, color=color_map.get(split_key, 'gray'), linewidth=2, linestyle='--', label=f'Sent ({split_key})')

    ax2.set_ylabel('Sentiment')
    title_symbol = getattr(bundle, 'symbol', 'Symbol')
    plt.title(f"{title_symbol} - Price vs Sentiment by Split (EWM)")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ---- Classification Plots (extracted from evaluator) ----

def plot_confusion_matrices(results: dict, save_path: str) -> None:
    """Plot confusion matrices for up to 4 models.
    results: { model_name: { 'confusion_matrix': np.ndarray, ... } }
    """
    n_models = min(len(results), 4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (model_name, metrics) in enumerate(results.items()):
        if i >= 4:
            break
        cm = metrics.get('confusion_matrix')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{model_name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        if cm is not None and cm.shape == (2, 2):
            axes[i].set_xticklabels(['Down', 'Up'])
            axes[i].set_yticklabels(['Down', 'Up'])

    for i in range(n_models, 4):
        axes[i].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(baseline_model: str, results: dict, baseline_results: dict | None, save_path: str, title: str) -> None:
    """Grouped bar chart comparing Accuracy/Precision/Recall/F1 across models.
    """
    models = list(results.keys())
    if baseline_results:
        models.append(f'{baseline_model}_Baseline')

    metrics_data = {
        'Accuracy': [results[m]['accuracy'] for m in results.keys()] + ([baseline_results['accuracy']] if baseline_results else []),
        'Precision': [results[m]['precision'] for m in results.keys()] + ([baseline_results['precision']] if baseline_results else []),
        'Recall': [results[m]['recall'] for m in results.keys()] + ([baseline_results['recall']] if baseline_results else []),
        'F1-Score': [results[m]['f1_score'] for m in results.keys()] + ([baseline_results['f1_score']] if baseline_results else []),
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(models))
    width = 0.2
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax.bar([xi + i * width for xi in x], values, width, label=metric, alpha=0.8)

    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks([xi + width * 1.5 for xi in x])
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_sentiment_impact(baseline_model: str, results: dict, baseline_results: dict, save_path: str, title: str) -> None:
    """Two side-by-side bars for accuracy and F1 comparing sentiment vs baseline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    models = [f'{baseline_model} + Sentiment', f'{baseline_model} Baseline']
    colors = ['#2E8B57', '#DC143C']

    accuracies = [results[baseline_model]['accuracy'], baseline_results['accuracy']]
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.7)
    ax1.set_title('Accuracy: Sentiment vs Baseline')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', va='bottom')

    f1_scores = [results[baseline_model]['f1_score'], baseline_results['f1_score']]
    bars2 = ax2.bar(models, f1_scores, color=colors, alpha=0.7)
    ax2.set_title('F1-Score: Sentiment vs Baseline')
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(0, 1)
    for bar, f1 in zip(bars2, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{f1:.3f}', ha='center', va='bottom')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sentiment_vs_predictions(bundle, y_true_binary, y_pred_binary, ticker, save_path):
    """
    Plot sentiment time series alongside model predictions (up/down) for classification.
    Shows sentiment_mean vs actual/predicted price movement direction.
    """
    # Combine all splits if available
    parts = []
    if hasattr(bundle, 'test_df') and bundle.test_df is not None and 'sentiment_mean' in bundle.test_df.columns:
        df = bundle.test_df.copy()
        parts.append(df)
    
    if not parts:
        print("Warning: No sentiment data found for plotting")
        return
    
    df_all = pd.concat(parts, axis=0)
    try:
        df_all.index = pd.to_datetime(df_all.index)
        df_all = df_all.sort_index()
    except Exception:
        pass
    
    # Align predictions with dates (take last N dates matching predictions)
    if len(df_all) >= len(y_pred_binary):
        df_plot = df_all.tail(len(y_pred_binary)).copy()
        df_plot['predicted'] = y_pred_binary
        if len(df_plot) >= len(y_true_binary):
            df_plot = df_plot.tail(len(y_true_binary))
            df_plot['actual'] = y_true_binary
        
        # Debug: Check prediction distribution
        unique_preds = np.unique(df_plot['predicted'])
        unique_actuals = np.unique(df_plot['actual'])
        pred_up_ratio = np.mean(df_plot['predicted'])
        actual_up_ratio = np.mean(df_plot['actual'])
        print(f"\n[DEBUG] Sentiment vs Predictions Plot:")
        print(f"  Predictions: unique={unique_preds}, Up ratio={pred_up_ratio:.3f}")
        print(f"  Actual: unique={unique_actuals}, Up ratio={actual_up_ratio:.3f}")
        print(f"  Date range: {df_plot.index.min()} to {df_plot.index.max()}")
        print(f"  Total samples: {len(df_plot)}")
    else:
        print("Warning: Date range mismatch for plotting")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot 1: Sentiment over time (FinBERT) + append price on secondary axis
    if 'sentiment_mean' in df_plot.columns:
        line_sent = ax1.plot(
            df_plot.index,
            df_plot['sentiment_mean'],
            'b-',
            linewidth=1.5,
            label='FinBERT Sentiment Mean',
            alpha=0.7,
        )
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Sentiment Score (FinBERT)', fontsize=11)
        ax1.set_title(f'{ticker} - FinBERT Sentiment vs Price Direction Predictions', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Secondary axis for Close price if available
        if 'close' in df_plot.columns:
            ax1b = ax1.twinx()
            line_price = ax1b.plot(
                df_plot.index,
                df_plot['close'],
                color='tab:green',
                linewidth=1.2,
                alpha=0.7,
                label='Close Price',
            )
            ax1b.set_ylabel('Price', fontsize=11, color='tab:green')
            ax1b.tick_params(axis='y', labelcolor='tab:green')

            # Build a combined legend for sentiment + price
            lines = line_sent + line_price
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
        else:
            ax1.legend(loc='upper left')
    
    # Plot 2: Actual vs Predicted (Up=1, Down=0)
    dates = df_plot.index
    ax2.plot(
        dates,
        df_plot['actual'],
        'g-',
        linewidth=2,
        label='Actual Direction (Price)',
        alpha=0.7,
        marker='o',
        markersize=3,
    )
    ax2.plot(
        dates,
        df_plot['predicted'],
        'r--',
        linewidth=2,
        label='Predicted Direction (Model)',
        alpha=0.7,
        marker='s',
        markersize=3,
    )
    ax2.set_ylabel('Direction (Up=1, Down=0)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylim([-0.1, 1.1])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Down', 'Up'])
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()