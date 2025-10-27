# https://medium.com/@wl8380/how-to-get-historic-stock-news-for-free-with-python-a-step-by-step-guide-02276d3c4860
# Source code: https://colab.research.google.com/drive/1OzKKKco-hx7CoH_uP1M2L7FsuLkcfu9l#scrollTo=yq064Ti9D2oL

import feedparser
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import os

def popular_companies(ticker):
    # Normalize ticker and safely strip exchange suffixes like '.AX'
    ticker = str(ticker).upper()
    # Split at first dot to remove suffixes (e.g., 'CBA.AX' -> 'CBA')
    if '.' in ticker:
        ticker = ticker.split('.', 1)[0]

    mapping = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "AMZN": "Amazon",
        "GOOGL": "Google",
        "NVDA": "NVIDIA",
        "TSLA": "Tesla",
        "META": "Meta",
        "JPM": "JPMorgan Chase",
        "BAC": "Bank of America",
        "V": "Visa",
        "MA": "Mastercard",
        "PYPL": "PayPal",
        "CBA": "Commonwealth Bank of Australia",
        "NAB": "National Australia Bank",
        "WBC": "Westpac Banking Corporation",
        "ANZ": "ANZ Banking Group",
        "BHP": "BHP Billiton",
        "RIO": "Rio Tinto",
        "WES": "Westpac Banking Corporation",
        "TLS": "Telstra Corporation Limited",
        "MQG": "Mitsubishi Corporation",
        "CSL": "CSL Limited",
    }
    # Fallback to using the ticker itself if not found
    return mapping.get(ticker, ticker)

# Fetch News
def _cache_path_for(ticker: str) -> str:
    return f"cache/sentiment/{ticker}_news.csv"


def _fetch_news_range(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch news for [start_date, end_date] inclusive via Google RSS."""
    company_name = popular_companies(ticker)
    query = quote_plus(
        f'{company_name} OR {ticker} after:{start_date.strftime("%Y-%m-%d")} before:{end_date.strftime("%Y-%m-%d")}'
    )
    rss_url = f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'
    feed = feedparser.parse(rss_url)

    rows = []
    for entry in feed.entries:
        try:
            if hasattr(entry, 'published_parsed'):
                parsed_date = datetime(*entry.published_parsed[:6])
                date_str = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
            else:
                date_str = 'N/A'
            rows.append({
                'date': date_str,
                'title': entry.title,
                'url': entry.link,
                'source': entry.get('source', {}).get('title', 'N/A'),
                'company': company_name,
                'ticker': ticker
            })
        except Exception as e:
            print(f"Error processing entry: {e}")
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')
    return df


def _read_cache(ticker: str) -> pd.DataFrame:
    from os.path import exists as _exists
    path = _cache_path_for(ticker)
    empty = pd.DataFrame(columns=['date','title','url','source','company','ticker'])
    if not _exists(path):
        return empty
    try:
        df = pd.read_csv(path)
        # If legacy cache had no header, re-read with names
        if 'date' not in df.columns or 'title' not in df.columns:
            df = pd.read_csv(path, header=None,
                        names=['date','title','url','source','company','ticker'])
    except Exception:
        return empty
    # Normalize types
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')
    # Ensure string cols exist
    for c in ['title','url','source','company','ticker']:
        if c not in df.columns:
            df[c] = ""
    return df


def _write_cache(ticker: str, df: pd.DataFrame) -> None:
    path = _cache_path_for(ticker)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.sort_values('date').to_csv(path, index=False)


def get_stock_news(ticker: str, start_date: datetime, end_date: datetime, verbose: bool = False) -> pd.DataFrame:
    """
    Cached news fetcher. If cache fully covers [start_date, end_date], return filtered.
    Otherwise fetch only the missing edges and append, then return filtered window.
    """
    cache_df = _read_cache(ticker)

    if not cache_df.empty:
        cached_start = cache_df['date'].min()
        cached_end = cache_df['date'].max()
    else:
        cached_start = None
        cached_end = None

    # Determine missing segments
    need_before = cached_start is None or start_date < cached_start
    need_after = cached_end is None or end_date > cached_end

    fetched_parts = []
    if need_before:
        fetch_end = (cached_start - timedelta(days=1)) if cached_start is not None else end_date
        fetch_start = start_date
        if fetch_end >= fetch_start:
            fetched_parts.append(_fetch_news_range(ticker, fetch_start, fetch_end))
    if need_after:
        fetch_start = (cached_end + timedelta(days=1)) if cached_end is not None else start_date
        fetch_end = end_date
        if fetch_end >= fetch_start:
            fetched_parts.append(_fetch_news_range(ticker, fetch_start, fetch_end))

    if fetched_parts:
        # Filter out empty DataFrames and prepare list for concatenation
        non_empty_parts = [part for part in fetched_parts if not part.empty]
        if cache_df is not None and not cache_df.empty:
            non_empty_parts.append(cache_df)
        
        # Only concatenate if we have non-empty DataFrames
        if non_empty_parts:
            new_df = pd.concat(non_empty_parts, axis=0, ignore_index=True)
        else:
            new_df = pd.DataFrame(columns=['date','title','url','source','company','ticker'])
        
        if not new_df.empty:
            # Deduplicate by URL then by (title,date)
            if 'url' in new_df.columns:
                new_df = new_df.drop_duplicates(subset=['url'])
            new_df = new_df.drop_duplicates(subset=['title','date'])
            new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
            new_df = new_df.dropna(subset=['date']).sort_values('date')
            _write_cache(ticker, new_df)
            cache_df = new_df
        if verbose:
            total_new = sum(len(p) for p in fetched_parts if not p.empty)
            print(f"  Appended {total_new} new articles to cache for {ticker}")
    else:
        if verbose:
            print(f"  Using cached news for {ticker}; range already covered")

    # Filter to requested window and return
    if not cache_df.empty:
        window_df = cache_df[(cache_df['date'] >= start_date) & (cache_df['date'] <= end_date)].copy()
    else:
        window_df = pd.DataFrame(columns=['date','title','url','source','company','ticker'])
    return window_df
