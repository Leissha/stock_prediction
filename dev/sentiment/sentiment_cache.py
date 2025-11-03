"""
Simplified sentiment cache management.
Handles sentiment analysis and caching efficiently.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

from sentiment.sentiment_analyzer import FinBERTAnalyzer
from sentiment.crawl_news import get_stock_news
from utils.color_log import cache, sentiment, social, warning


class SentimentCache:
    """
    Simplified sentiment cache management.
    Handles news fetching, sentiment analysis, and caching.
    """
    
    def __init__(self):
        self.analyzer = FinBERTAnalyzer()
    
    def get_news_with_sentiment(self, ticker: str, start_date: str, end_date: str, source: str = 'all', include_social: bool = False) -> pd.DataFrame:
        """
        Get news with sentiment analysis - single method that handles everything.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: News source ('all', 'google', 'yahoo', 'businesstoday'). Default: 'all'
            include_social: Whether to include social media data (Reddit, Google Trends). Default: False
            
        Returns:
            DataFrame with news and sentiment scores
        """
        # Simple path construction
        cache_path = Path("cache/sentiment")
        cache_path.mkdir(parents=True, exist_ok=True)
        file_path = cache_path / f"{ticker}_news.csv"
        
        # 1. Load existing cache
        cached_df = pd.DataFrame()
        if file_path.exists():
            cached_df = pd.read_csv(file_path)
            cached_df['date'] = pd.to_datetime(cached_df['date'])
            
            # Check if cache covers the requested range
            if self._cache_covers_range(cached_df, start_date, end_date):
                cache(f"Using cached sentiment data for {ticker}")
                
                # If include_social, check if Reddit cache is sufficient
                if include_social:
                    should_skip_reddit = self._check_social_cache_sufficient(cached_df, start_date, end_date, verbose=True)
                    if should_skip_reddit:
                        # Cache sufficient, return cached data (already includes Reddit)
                        return cached_df
                    # Otherwise, will fetch below to fill gaps
                
                # Regular cache hit (no social or social cache sufficient)
                return cached_df
        
        # 2. Fetch new articles (and optionally social media)
        sentiment(f"Fetching new articles for {ticker}")
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Only include_social if cache is insufficient (checked above)
        news_df = get_stock_news(ticker, start_dt, end_dt, verbose=True, source=source, include_social=include_social)
        
        if news_df.empty:
            return news_df
        
        # 3. Analyze sentiment for articles without scores
        missing_sentiment = 0
        if 'sentiment_score' in news_df.columns:
            missing_sentiment = news_df['sentiment_score'].isna().sum()
        
        if missing_sentiment > 0:
            sentiment(f"Analyzing sentiment for {missing_sentiment} articles...")
            missing_mask = news_df['sentiment_score'].isna()
            articles_to_analyze = news_df[missing_mask]
            
            if len(articles_to_analyze) > 0:
                sentiment_results = self.analyzer.analyze_batch(articles_to_analyze['title'].tolist())
                
                # Extract scores and labels
                sentiment_scores = [result['sentiment_score'] for result in sentiment_results]
                sentiment_labels = [int(result['predicted_class']) for result in sentiment_results]
                
                # Map numeric labels to text labels
                label_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'}
                sentiment_labels = [label_mapping[label] for label in sentiment_labels]
                
                # Update only the missing sentiment scores
                news_df.loc[missing_mask, 'sentiment_score'] = sentiment_scores
                news_df.loc[missing_mask, 'sentiment_label'] = sentiment_labels
        elif 'sentiment_score' not in news_df.columns:
            # No sentiment column exists, analyze all articles
            sentiment(f"Analyzing sentiment for {len(news_df)} articles...")
            sentiment_results = self.analyzer.analyze_batch(news_df['title'].tolist())
            
            # Extract scores and labels
            sentiment_scores = [result['sentiment_score'] for result in sentiment_results]
            sentiment_labels = [int(result['predicted_class']) for result in sentiment_results]
            
            # Map numeric labels to text labels
            label_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'}
            sentiment_labels = [label_mapping[label] for label in sentiment_labels]
            
            # Add sentiment scores to the data
            news_df = news_df.copy()
            news_df['sentiment_score'] = sentiment_scores
            news_df['sentiment_label'] = sentiment_labels
        
        # 4. Save to cache
        news_df.to_csv(file_path, index=False)
        cache(f"Sentiment data cached for {ticker}")
        
        return news_df

    def build_daily_sentiment(self, ticker: str, start_date: str, end_date: str, source: str = 'all', include_social: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build daily sentiment aggregation with rich features.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: News source ('all', 'google', 'yahoo', 'businesstoday'). Default: 'all'
            include_social: Whether to include social media data (Reddit, Google Trends). Default: False

        Returns:
            Tuple of (news_df, daily_sentiment_df with engineered features)
        """
        # Get news with sentiment (optionally include social media)
        news_df = self.get_news_with_sentiment(ticker, start_date, end_date, source=source, include_social=include_social)

        if news_df.empty:
            return news_df, pd.DataFrame()

        # Check if sentiment scores exist
        if 'sentiment_score' not in news_df.columns:
            warning(f"No sentiment scores found for {ticker}")
            return news_df, pd.DataFrame()

        # Filter out NaN sentiment scores before aggregation
        valid_news = news_df.dropna(subset=['sentiment_score']).copy()

        if len(valid_news) == 0:
            return news_df, pd.DataFrame()

        # Focus on temporal patterns: rolling windows, lags, momentum
        valid_news['date_only'] = valid_news['date'].dt.date
        grouped = valid_news.groupby('date_only')['sentiment_score']

        # Feature 1: Daily mean sentiment (baseline)
        daily_sentiment = grouped.agg(['mean', 'count']).reset_index()
        daily_sentiment.columns = ['date_only', 'sentiment_mean', 'news_count']

        # Sort by date for temporal features
        daily_sentiment = daily_sentiment.sort_values('date_only').reset_index(drop=True)

        # Feature 2: 3-day rolling average (short-term trend)
        daily_sentiment['sentiment_roll3'] = daily_sentiment['sentiment_mean'].rolling(window=3, min_periods=1).mean()

        # Feature 3: 7-day rolling average (medium-term trend)
        daily_sentiment['sentiment_roll7'] = daily_sentiment['sentiment_mean'].rolling(window=7, min_periods=1).mean()

        # Feature 4: 1-day lag (previous day sentiment)
        daily_sentiment['sentiment_lag1'] = daily_sentiment['sentiment_mean'].shift(1)

        # Feature 5: 3-day cumulative sentiment momentum (trend direction)
        daily_sentiment['sentiment_momentum3'] = daily_sentiment['sentiment_mean'].rolling(window=3, min_periods=1).apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0, raw=False
        )

        # Fill NaN values (first rows will be 0 for lag and momentum)
        daily_sentiment = daily_sentiment.fillna(0)

        # Set index for merging
        daily_sentiment = daily_sentiment.set_index('date_only')

        return news_df, daily_sentiment
    
    def integrate_sentiment(self, df: pd.DataFrame, ticker: str, start_date: str, end_date: str, source: str = 'all', include_social: bool = False) -> pd.DataFrame:
        """
        Integrate sentiment features into stock data.
        Time Alignment Strategy:
            1. Group news by calendar day (date_only)
            2. Merge with stock data (left join to preserve all trading days)
            3. Forward-fill sentiment for days without news (no backward fill to prevent data leakage)
            4. Fill remaining NaNs with 0 (neutral sentiment, no news)
        Args:
            df: Stock data DataFrame
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: News source ('all', 'google', 'yahoo', 'businesstoday'). Default: 'all'
            
        Returns:
            DataFrame with sentiment features added
        """
        # Get daily sentiment (optionally including social media)
        news_df, daily_sentiment = self.build_daily_sentiment(ticker, start_date, end_date, source=source, include_social=include_social)
        
        if news_df.empty or daily_sentiment.empty:
            warning(f"No sentiment data found for {ticker}")
            return df
        
        # Merge with stock data
        df_with_sentiment = df.copy()
        
        # Handle different index types safely
        try:
            if hasattr(df_with_sentiment.index, 'date'):
                df_with_sentiment['date_only'] = df_with_sentiment.index.date  # type: ignore
            else:
                df_with_sentiment['date_only'] = pd.to_datetime(df_with_sentiment.index).date
        except (AttributeError, TypeError):
            df_with_sentiment['date_only'] = pd.to_datetime(df_with_sentiment.index).date
        
        # Merge sentiment data
        df_with_sentiment = df_with_sentiment.merge(
            daily_sentiment, 
            left_on='date_only', 
            right_index=True, 
            how='left'
        )
        
        # Get all sentiment column names (exclude date_only)
        sentiment_cols = [col for col in daily_sentiment.columns if col != 'date_only']

        # Forward-fill missing sentiment for all features
        df_with_sentiment[sentiment_cols] = df_with_sentiment[sentiment_cols].ffill().fillna(0)

        # Drop temporary column
        df_with_sentiment = df_with_sentiment.drop('date_only', axis=1)

        # Count articles with valid sentiment scores
        articles_with_sentiment = news_df['sentiment_score'].notna().sum() if 'sentiment_score' in news_df.columns else 0

        sentiment(f"Added {len(sentiment_cols)} sentiment features")
        sentiment(f"Total news articles fetched: {len(news_df)}")
        sentiment(f"Articles with sentiment: {articles_with_sentiment}")
        sentiment(f"Days with news: {len(daily_sentiment)}")

        return df_with_sentiment
    
    def _cache_covers_range(self, cached_df: pd.DataFrame, start_date: str, end_date: str) -> bool:
        """
        Check if cached data covers the requested date range.
        
        Args:
            cached_df: Cached DataFrame
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            True if cache covers the range, False otherwise
        """
        if cached_df.empty:
            return False
        
        cached_start = cached_df['date'].min()
        cached_end = cached_df['date'].max()
        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)
        
        return cached_start <= requested_start and cached_end >= requested_end
    
    def _check_social_cache_sufficient(self, cached_df: pd.DataFrame, start_date: str, end_date: str, verbose: bool = False) -> bool:
        """
        Check if social media (Reddit, Google Trends) cache is sufficient to skip fetching.
        
        Args:
            cached_df: Full cached DataFrame
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            verbose: Print debug messages
            
        Returns:
            True if cache is sufficient (should skip fetch), False otherwise
        """
        from utils.color_log import social
        
        if cached_df.empty:
            return False
        
        # Get social media posts from cache
        cached_social = cached_df[cached_df["source"].str.contains("Reddit|Google Trends", case=False, na=False)].copy()
        
        if cached_social.empty:
            return False
        
        # Filter to requested date range
        cached_social['date'] = pd.to_datetime(cached_social['date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        social_in_range = cached_social[(cached_social["date"] >= start_dt) & (cached_social["date"] <= end_dt)]
        
        if social_in_range.empty:
            return False
        
        # Calculate coverage: how many days in the range have cached posts
        cached_dates = social_in_range['date'].dt.date
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        date_range_set = set(date_range.date)
        cached_dates_set = set(cached_dates)
        
        coverage = len(cached_dates_set & date_range_set) / len(date_range_set) if len(date_range_set) > 0 else 0
        
        # For Reddit: Skip fetching if we have substantial cached data (>100 posts) and decent coverage (>=40%)
        # Coverage threshold lowered to 40% because Reddit posts cluster on news/event days
        # Eg, we skipped fetching for AAPL news with 588 posts covering 349/731 days (47.74%)
        is_sufficient = len(social_in_range) > 100 and coverage >= 0.40
        
        if verbose:
            if is_sufficient:
                social(f"Using {len(social_in_range)} cached social media posts (coverage: {coverage*100:.1f}%, {len(cached_dates_set)}/{len(date_range_set)} days, skipping fetch)")
            else:
                social(f"Social cache insufficient: {len(social_in_range)} posts, {coverage*100:.1f}% coverage. Will fetch to fill gaps.")
        
        return is_sufficient


# Global sentiment cache instance
sentiment_cache = SentimentCache()


def get_sentiment_cache() -> SentimentCache:
    """Get the global sentiment cache instance."""
    return sentiment_cache