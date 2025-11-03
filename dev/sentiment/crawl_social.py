"""
Social Media Data Sources for Stock Sentiment Analysis

Based on real-world fintech implementations:
- Reddit (r/wallstreetbets, r/investing, r/stocks) - FREE via PRAW
- Google Trends - FREE, search volume trends
- Twitter/X - Expensive ($5K/month) no thank you, next!

Reference:
https://medium.datadriveninvestor.com/predicting-market-sentiment-with-social-media-a-deep-learning-approach-to-fintech-trading-91993eca0af4
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv
import praw
from pytrends.request import TrendReq

load_dotenv(override=True)


from sentiment.crawl_news import OUTPUT_COLUMNS, _ticker_clean, popular_companies
from utils.color_log import social, warning, error, success

# Reddit subreddits for stock discussions
STOCK_SUBREDDITS = [
    "wallstreetbets",      # Retail trading, high volume, meme stocks
    "investing",           # Long-term investors, fundamental analysis
    "stocks",              # General stock discussions
    "StockMarket",         # Market news and analysis
    "SecurityAnalysis",    # Fundamental analysis, value investing
    "ASX",                 # Australian Stock Exchange discussions
    "AusFinance",          # Australian finance and investing
]

# Rate limiting
REDDIT_RATE_LIMIT = 60  # requests per minute

def _setup_reddit_client() -> Optional[praw.Reddit]:
    """
    Setup Reddit API client using PRAW.
    
    Steps:
    1. Go to https://www.reddit.com/prefs/apps
    2. Click "create app" or "create another app"
    3. Choose "script" type
    4. Note: client_id (under app name), secret (next to "secret")
    5. Set environment variables:
       - REDDIT_CLIENT_ID
       - REDDIT_CLIENT_SECRET
       - REDDIT_USER_AGENT (e.g., "sentiment-bot/1.0 by /u/yourusername")
    
    Free tier allows:
    - 60 requests per minute
    - Up to 250 results per search
    - No credit card required
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "sentiment-bot/1.0")
    
    # Strip quotes if present (dotenv sometimes includes them)
    if client_id:
        client_id = client_id.strip().strip('"').strip("'")
    if client_secret:
        client_secret = client_secret.strip().strip('"').strip("'")
    
    if not client_id or not client_secret:
        warning("Reddit credentials not found in .env file")
        warning("Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in dev/.env")
        warning("See docs/SOCIAL_MEDIA_SETUP.md for setup instructions")
        return None
    
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        # Test connection
        reddit.read_only = True
        _ = reddit.subreddit("test").id  # Quick test
        return reddit
    except Exception as e:
        error(f"Failed to setup Reddit client: {e}")
        return None


def fetch_reddit_posts(ticker: str, start_date: datetime, end_date: datetime, 
                       subreddits: Optional[List[str]] = None, 
                       limit: int = 250) -> pd.DataFrame:
    """
    Fetch Reddit posts mentioning a ticker from specified subreddits.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date for search
        end_date: End date for search
        subreddits: List of subreddit names (default: STOCK_SUBREDDITS)
        limit: Max posts per subreddit (default: 250, Reddit API limit)
    
    Returns:
        DataFrame with columns: date, title, url, source, original_source, company, ticker
        Note: 'title' contains post title + selftext, 'url' is post permalink
    
    Example:
        >>> reddit = _setup_reddit_client()
        >>> posts = fetch_reddit_posts("AAPL", datetime(2024,1,1), datetime(2024,1,31))
    """
    if subreddits is None:
        subreddits = STOCK_SUBREDDITS
    
    reddit = _setup_reddit_client()
    if reddit is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    ticker_upper = _ticker_clean(ticker)
    company_name = popular_companies(ticker)
    rows = []
    
    # Build search query: ticker, $ticker, and company name
    # This helps find posts that mention the company by name instead of ticker
    search_terms = [ticker_upper, f"${ticker_upper}"]
    if company_name and company_name.upper() != ticker_upper:
        # Only add company name if it's different from ticker (avoid duplicates)
        search_terms.append(company_name)
    
    query = " OR ".join(search_terms)
    
    total_subreddits = len(subreddits)
    for idx, subreddit_name in enumerate(subreddits, 1):
        try:
            social(f"Searching r/{subreddit_name} ({idx}/{total_subreddits})...")
            subreddit = reddit.subreddit(subreddit_name)
            
            try:
                # Use search API which is more efficient than fetching all hot posts
                # Search by 'new' with time filter to get historical posts
                posts = subreddit.search(query, sort='new', limit=limit, time_filter='all')
                
                posts_processed = 0
                for post in posts:
                    # Check date range first (early exit for efficiency)
                    post_date = datetime.fromtimestamp(post.created_utc)
                    if post_date < start_date or post_date > end_date:
                        continue
                    
                    # Check if post mentions ticker or company name (title or selftext)
                    content = f"{post.title} {post.selftext}".upper()
                    content_lower = f"{post.title} {post.selftext}".lower()
                    
                    # Check for ticker mentions
                    has_ticker = (ticker_upper in content or f"${ticker_upper}" in content)
                    # Check for company name mentions
                    has_company = company_name and (company_name.upper() in content or company_name.lower() in content_lower)
                    
                    if not (has_ticker or has_company):
                        continue
                    
                    # Combine title and selftext for sentiment analysis
                    text_content = post.title
                    if post.selftext:
                        text_content += f" {post.selftext[:500]}"  # Limit length
                    
                    rows.append({
                        "date": post_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "title": text_content,
                        "url": f"https://www.reddit.com{post.permalink}",
                        "source": f"Reddit/r/{subreddit_name}",
                        "original_source": f"Reddit/r/{subreddit_name}",
                        "company": ticker_upper,
                        "ticker": ticker,
                    })
                    posts_processed += 1
                
                if posts_processed > 0:
                    success(f"Found {posts_processed} posts in r/{subreddit_name}")
                
            except Exception as e:
                warning(f"Error searching r/{subreddit_name}: {e}")
                continue
            
            # Rate limiting: sleep between subreddit searches (not per-post)
            # This respects Reddit's 60 requests/minute limit
            if idx < total_subreddits:  # Don't sleep after last subreddit
                time.sleep(60 / REDDIT_RATE_LIMIT)  # 1 second between subreddits
                    
        except Exception as e:
            warning(f"Error accessing subreddit r/{subreddit_name}: {e}")
            continue
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=['url', 'date'])
        df = df.sort_values('date')
    
    return df


def fetch_google_trends(ticker: str, start_date: datetime, end_date: datetime, verbose: bool = False) -> pd.DataFrame:
    """
    Fetch Google Trends search volume data for a ticker.
    
    Google Trends (FREE):
    - No API key required
    - Historical data available (up to 5 years)
    - Returns relative search interest (0-100 scale)
    - Can detect spikes in public interest
    
    Use Case:
    - Spike in search volume may correlate with price movements
    - Can be used as a sentiment proxy (high search = high interest)
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with daily search interest scores
        Note: This returns TREND data, not text content
    """
    ticker_clean = _ticker_clean(ticker)
    
    try:
        # Initialize with retry parameters (Google Trends can be rate-limited)
        pytrends = TrendReq(hl='en-US', tz=360, retries=2, backoff_factor=0.1) # type: ignore
        
        # Build keyword list (ticker + "$TICKER" format)
        keywords = [ticker_clean, f"${ticker_clean}"]
        
        # Fetch trends with proper rate limiting
        # Google Trends allows ~5 requests/minute, so we wait before each request
        # But usually we just need to run once in 1 minute so not a big prob
        time.sleep(2)  # Wait 2 seconds before making request (respectful rate limiting)
        
        timeframe_str = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
        pytrends.build_payload(keywords, timeframe=timeframe_str, geo='US')
        trends_df = pytrends.interest_over_time()
        
        if trends_df.empty:
            warning(f"Google Trends returned no data for {ticker_clean} (may be normal if search volume is too low)")
            return pd.DataFrame()
        
        # Check if data has 'isPartial' column (indicates incomplete data)
        if 'isPartial' in trends_df.columns:
            trends_df = trends_df[trends_df['isPartial'] == False]
        
        if trends_df.empty:
            warning(f"Google Trends data for {ticker_clean} contains only partial data")
            return pd.DataFrame()
        
        # Convert to standard format
        rows = []
        for date_idx, row in trends_df.iterrows():
            # Average interest score across keywords (skip isPartial if present)
            numeric_cols = [col for col in keywords if col in row.index]
            if numeric_cols:
                avg_interest = row[numeric_cols].mean()
            else:
                continue  # Skip if no numeric data
            
            # Ensure date is datetime for strftime
            if isinstance(date_idx, pd.Timestamp):
                date_str = date_idx.strftime("%Y-%m-%d %H:%M:%S")
            else:
                date_str = str(date_idx)
            
            rows.append({
                "date": date_str,
                "title": f"Google Trends Interest: {avg_interest:.1f}",
                "url": f"https://trends.google.com/trends/explore?q={ticker_clean}",
                "source": "Google Trends",
                "original_source": "Google Trends",
                "company": ticker_clean,
                "ticker": ticker,
                # Store interest score in title for later parsing
                "_trend_score": avg_interest,
            })
        
        if verbose and rows:
            success(f"Retrieved {len(rows)} Google Trends data points for {ticker_clean}")
        
        return pd.DataFrame(rows)
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "TooManyRequests" in error_msg:
            warning(f"Google Trends rate limited for {ticker}. Please wait a few minutes before retrying.")
            warning("Google Trends allows ~5 requests/minute. Consider caching results.")
        else:
            warning(f"Google Trends error for {ticker}: {e}")
        return pd.DataFrame()


def get_social_media_data(ticker: str, start_date: datetime, end_date: datetime,
                          sources: Optional[List[str]] = None, verbose: bool = False) -> pd.DataFrame:
    """
    Unified function to fetch social media data from multiple sources.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data collection
        end_date: End date for data collection
        sources: List of sources to fetch from
                Options: ['reddit', 'trends', 'all']
        verbose: Print progress messages
    
    Returns:
        Combined DataFrame from all requested sources
    
    Example:
        >>> df = get_social_media_data("AAPL", datetime(2024,1,1), datetime(2024,1,31), sources=['reddit', 'trends'])
    """
    if sources is None:
        sources = ['reddit']  # Default to Reddit (most reliable)
    
    if 'all' in sources:
        sources = ['reddit', 'trends']
    
    all_data = []
    
    if 'reddit' in sources:
        if verbose:
            social(f"Fetching Reddit posts for {ticker}...")
        reddit_df = fetch_reddit_posts(ticker, start_date, end_date)
        if not reddit_df.empty:
            all_data.append(reddit_df)
            if verbose:
                success(f"Found {len(reddit_df)} Reddit posts")
    
    if 'trends' in sources:
        if verbose:
            social(f"Fetching Google Trends data for {ticker}...")
        trends_df = fetch_google_trends(ticker, start_date, end_date, verbose=verbose)
        if not trends_df.empty:
            all_data.append(trends_df)
        # Note: fetch_google_trends already prints success message if verbose=True
    
    if not all_data:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    # Merge and deduplicate
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['title', 'date', 'source'])
    combined = combined.sort_values('date')
    
    return combined