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


class SentimentCache:
    """
    Simplified sentiment cache management.
    Handles news fetching, sentiment analysis, and caching.
    """
    
    def __init__(self):
        self.analyzer = FinBERTAnalyzer()
    
    def get_news_with_sentiment(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get news with sentiment analysis - single method that handles everything.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with news and sentiment scores
        """
        # Simple path construction
        cache_path = Path("cache/sentiment")
        cache_path.mkdir(parents=True, exist_ok=True)
        file_path = cache_path / f"{ticker}_news.csv"
        
        # 1. Load existing cache
        if file_path.exists():
            cached_df = pd.read_csv(file_path)
            cached_df['date'] = pd.to_datetime(cached_df['date'])
            
            # Check if cache covers the requested range
            if self._cache_covers_range(cached_df, start_date, end_date):
                print(f"Using cached sentiment data for {ticker}")
                return cached_df
        
        # 2. Fetch new articles
        print(f"Fetching new articles for {ticker}")
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        news_df = get_stock_news(ticker, start_dt, end_dt, verbose=True)
        
        if news_df.empty:
            return news_df
        
        # 3. Analyze sentiment for articles without scores
        missing_sentiment = 0
        if 'sentiment_score' in news_df.columns:
            missing_sentiment = news_df['sentiment_score'].isna().sum()
        
        if missing_sentiment > 0:
            print(f"Analyzing sentiment for {missing_sentiment} articles...")
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
            print(f"Analyzing sentiment for {len(news_df)} articles...")
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
        print(f"Sentiment data cached for {ticker}")
        
        return news_df
    
    def build_daily_sentiment(self, ticker: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build daily sentiment aggregation.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Tuple of (news_df, daily_sentiment_df)
        """
        # Get news with sentiment
        news_df = self.get_news_with_sentiment(ticker, start_date, end_date)
        
        if news_df.empty:
            return news_df, pd.DataFrame()
        
        # Check if sentiment scores exist
        if 'sentiment_score' not in news_df.columns:
            print(f"No sentiment scores found for {ticker}")
            return news_df, pd.DataFrame()
        
        # Build daily aggregated sentiment
        news_df['date_only'] = news_df['date'].dt.date
        
        # Filter out NaN sentiment scores before aggregation
        valid_news = news_df.dropna(subset=['sentiment_score'])
        
        if len(valid_news) > 0:
            daily_sentiment = valid_news.groupby('date_only')['sentiment_score'].agg(['mean', 'count']).fillna(0)
            daily_sentiment.columns = ['sentiment_mean', 'news_count']
        else:
            # No valid sentiment data
            daily_sentiment = pd.DataFrame(columns=['sentiment_mean', 'news_count'])
        
        return news_df, daily_sentiment
    
    def integrate_sentiment(self, df: pd.DataFrame, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Integrate sentiment features into stock data.
        
        Args:
            df: Stock data DataFrame
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with sentiment features added
        """
        # Get daily sentiment
        news_df, daily_sentiment = self.build_daily_sentiment(ticker, start_date, end_date)
        
        if news_df.empty or daily_sentiment.empty:
            print(f"No sentiment data found for {ticker}")
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
        
        # Forward-fill missing sentiment
        sentiment_cols = ['sentiment_mean', 'news_count']
        df_with_sentiment[sentiment_cols] = df_with_sentiment[sentiment_cols].ffill().fillna(0)
        
        # Drop temporary column
        df_with_sentiment = df_with_sentiment.drop('date_only', axis=1)
        
        print(f"Added {len(sentiment_cols)} sentiment features")
        print(f"News articles processed: {len(news_df)}")
        print(f"Days with news: {len(daily_sentiment)}")
        
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


# Global sentiment cache instance
sentiment_cache = SentimentCache()


def get_sentiment_cache() -> SentimentCache:
    """Get the global sentiment cache instance."""
    return sentiment_cache

