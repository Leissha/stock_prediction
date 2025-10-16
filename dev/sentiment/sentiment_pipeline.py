import pandas as pd
import numpy as np
from sentiment.crawl_news import get_stock_news
from sentiment.sentiment_analyzer import FinBERTAnalyzer

def build_daily_sentiment(ticker: str, start_date: str, end_date: str):
    """
    Build raw news and aggregated daily sentiment for a ticker and date range.

    Returns:
        news_df (pd.DataFrame): raw news with per-article sentiment_score
        daily_sentiment (pd.DataFrame): daily aggregated sentiment metrics
    """
    from datetime import datetime

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Delegate logging to the crawler (verbose prints whether cache is used or new items appended)
    news_df = get_stock_news(ticker, start_dt, end_dt, verbose=True)
    if news_df.empty:
        return news_df, pd.DataFrame()

    print("  Analyzing sentiment with FinBERT...")
    analyzer = FinBERTAnalyzer()
    news_df['sentiment'] = news_df['title'].apply(
        lambda text: analyzer.analyze_text(text)['sentiment_score']
    )

    news_df['date_only'] = news_df['date'].dt.date
    daily_sentiment = news_df.groupby('date_only')['sentiment'].agg([
        'mean', 'median', 'std', 'count'
    ]).fillna(0)
    daily_sentiment.columns = ['s_mean', 's_median', 's_std', 'news_count']
    daily_sentiment['pos_ratio'] = (
        news_df.groupby('date_only')['sentiment'].apply(lambda x: (x > 0.1).mean())
    ).fillna(0)
    daily_sentiment['neg_ratio'] = (
        news_df.groupby('date_only')['sentiment'].apply(lambda x: (x < -0.1).mean())
    ).fillna(0)
    daily_sentiment['entropy'] = (
        news_df.groupby('date_only')['sentiment'].apply(
            lambda x: -sum(p * np.log2(p + 1e-10) for p in [x[x > 0.1].count()/len(x),
                                                           x[x < -0.1].count()/len(x),
                                                           x[(x >= -0.1) & (x <= 0.1)].count()/len(x)]
                          if p > 0)
        )
    ).fillna(0)

    return news_df, daily_sentiment

def integrate_news_sentiment(df: pd.DataFrame, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Integrate news sentiment analysis into the pipeline using crawl_news logic.
    
    Args:
        df: Stock price DataFrame
        ticker: Stock ticker symbol
        start_date: Start date for news fetching
        end_date: End date for news fetching
    
    Returns:
        DataFrame with sentiment features added
    """
    # Reuse builder to keep logic consistent
    news_df, daily_sentiment = build_daily_sentiment(ticker, start_date, end_date)
    
    if news_df.empty:
        print(f"No news found for {ticker}")
        return df
    
    # If no news, return original df
    if news_df.empty or daily_sentiment.empty:
        print(f"No news found for {ticker}")
        return df
    
    # Merge with stock data
    df_with_sentiment = df.copy()
    # Handle different index types safely
    try:
        if hasattr(df_with_sentiment.index, 'date'):
            df_with_sentiment['date_only'] = df_with_sentiment.index.date  # type: ignore
        else:
            df_with_sentiment['date_only'] = pd.to_datetime(df_with_sentiment.index).date  # type: ignore
    except (AttributeError, TypeError):
        df_with_sentiment['date_only'] = pd.to_datetime(df_with_sentiment.index).date  # type: ignore
    
    # Merge sentiment data
    df_with_sentiment = df_with_sentiment.merge(
        daily_sentiment, 
        left_on='date_only', 
        right_index=True, 
        how='left'
    )
    
    # Interpolate missing sentiment values over time, then fill any remaining gaps with 0
    sentiment_cols = ['s_mean', 's_median', 's_std', 'news_count', 'pos_ratio', 'neg_ratio', 'entropy']
    # Ensure datetime index for time-based interpolation
    if not isinstance(df_with_sentiment.index, pd.DatetimeIndex):
        try:
            df_with_sentiment.index = pd.to_datetime(df_with_sentiment.index)
        except Exception:
            pass
    # Sort for stable interpolation
    try:
        df_with_sentiment = df_with_sentiment.sort_index()
    except Exception:
        pass
    # Interpolate linearly (or time if DatetimeIndex), then fill remaining NaNs with 0
    try:
        interp_method = 'time' if isinstance(df_with_sentiment.index, pd.DatetimeIndex) else 'linear'
        df_with_sentiment[sentiment_cols] = (
            df_with_sentiment[sentiment_cols]
            .interpolate(method=interp_method, limit_direction='both')
            .fillna(0)
        )
    except Exception:
        # Fallback to simple fill if interpolation fails
        df_with_sentiment[sentiment_cols] = df_with_sentiment[sentiment_cols].fillna(0)
    
    # Drop the temporary date_only column
    df_with_sentiment = df_with_sentiment.drop('date_only', axis=1)
    
    print(f"  Added {len(sentiment_cols)} sentiment features to DataFrame")
    print(f"   News articles processed: {len(news_df)}")
    print(f"   Days with news: {len(daily_sentiment)}")
    print(f"   Sentiment columns: {sentiment_cols}")
    
    return df_with_sentiment
