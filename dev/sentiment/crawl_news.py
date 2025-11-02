"""
News crawling utilities (Google News RSS, Yahoo Finance RSS, Business Today scraper)
with caching, de-duplication and consistent output schema.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from typing import List

import feedparser
import pandas as pd
import requests as req
from bs4 import BeautifulSoup as BS  # type: ignore
from utils.color_log import cache, info


# --- Constants -----------------------------------------------------------------

USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; sentiment-bot/1.0)"}
PROVIDER_PRIORITY = {
    "Yahoo Finance": 0,
    "Business Today": 1,
    "Google News": 2,
}
OUTPUT_COLUMNS = [
    "date",
    "title",
    "url",
    "source",
    "original_source",
    "company",
    "ticker",
]


def _cache_path_for(ticker: str) -> str:
    return f"cache/sentiment/{ticker}_news.csv"


def _ticker_clean(ticker: str) -> str:
    t = str(ticker).upper()
    return t.split(".", 1)[0] if "." in t else t


def popular_companies(ticker: str) -> str:
    t = _ticker_clean(ticker)
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
        "WES": "Wesfarmers",
        "TLS": "Telstra Corporation Limited",
        "MQG": "Macquarie Group",
        "CSL": "CSL Limited",
    }
    return mapping.get(t, t)


def _merge_and_dedupe(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Prefer provider-specific sources for identical URLs
    if {"url", "source"}.issubset(df.columns):
        df["__src_priority"] = df["source"].map(PROVIDER_PRIORITY).fillna(9)
        df = df.sort_values(["url", "__src_priority"])
        df = df.drop(columns=["__src_priority"])  # keep sorting influence only
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")
    df = df.drop_duplicates(subset=["title", "date"])
    df = df.dropna(subset=["date"]).sort_values("date")
    # Ensure schema
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col != "original_source" else "N/A"
    return df[OUTPUT_COLUMNS]


# --- Providers -----------------------------------------------------------------

def _fetch_news_yahoo(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    candidates = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://finance.yahoo.com/rss/headline?s={ticker}",
    ]
    t_clean = _ticker_clean(ticker)
    if t_clean != ticker:
        candidates += [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t_clean}&region=US&lang=en-US",
            f"https://finance.yahoo.com/rss/headline?s={t_clean}",
        ]

    company_name = popular_companies(ticker)
    rows = []
    for rss_url in candidates:
        try:
            feed = feedparser.parse(rss_url, request_headers=USER_AGENT)
            for entry in getattr(feed, "entries", []):
                try:
                    if hasattr(entry, "published_parsed"):
                        d = datetime(*entry.published_parsed[:6])
                        if start_date <= d <= end_date:
                            rows.append({
                                "date": d.strftime("%Y-%m-%d %H:%M:%S"),
                                "title": entry.title,
                                "url": getattr(entry, "link", entry.get("link", "N/A")),
                                "source": "Yahoo Finance",
                                "original_source": entry.get("source", {}).get("title", "N/A"),
                                "company": company_name,
                                "ticker": ticker,
                            })
                except Exception:
                    continue
            if rows:
                break
        except Exception:
            continue
    return _merge_and_dedupe([pd.DataFrame(rows)])


def _fetch_news_businesstoday(ticker: str, start_date: datetime, end_date: datetime, verbose: bool = False) -> pd.DataFrame:
    url = "https://www.businesstoday.in/latest/economy"
    try:
        r = req.get(url, headers=USER_AGENT, timeout=15)
        r.raise_for_status()
    except Exception as e:
        if verbose:
            print(f"  Error fetching Business Today: {e}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    trav = BS(r.content, "html.parser")
    rows = []
    seen = set()
    now_utc = datetime.utcnow()
    company_name = popular_companies(ticker)
    t_clean = _ticker_clean(ticker)
    terms = {company_name.lower(), t_clean.lower()}

    from bs4.element import Tag

    for link in trav.find_all("a"):
        try:
            if not isinstance(link, Tag):
                continue
            title_text = link.get_text(strip=True)
            if not title_text or len(title_text) <= 35:
                continue
            txt_lower = title_text.lower()
            matches_company = any(term in txt_lower for term in terms)
            # Keep general economy news too (no hard filter), but prefer company matches
            href_val = link.get("href")
            href = href_val if isinstance(href_val, str) else ""
            if not href:
                continue
            if isinstance(href, str) and href.startswith("/"):
                href = f"https://www.businesstoday.in{href}"
            key = (title_text, href)
            if key in seen:
                continue
            seen.add(key)

            # Approximate timestamp
            if start_date <= now_utc <= end_date:
                rows.append({
                    "date": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "title": title_text,
                    "url": href,
                    "source": "Business Today",
                    "original_source": "Business Today",
                    "company": company_name,
                    "ticker": ticker,
                })
        except Exception:
            continue

    df = _merge_and_dedupe([pd.DataFrame(rows)])
    if not df.empty:
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    return df


def _fetch_news_google(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    company_name = popular_companies(ticker)
    query = quote_plus(
        f"{company_name} OR {ticker} after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}"
    )
    
    # Detect Australian stocks and use AU locale for better coverage
    # CBA.AX, WBC.AX, etc. will get Australian news sources
    is_australian = ticker.endswith('.AX') or ticker.upper().endswith('.AX')
    
    if is_australian:
        # Australian locale for better local news coverage
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-AU&gl=AU&ceid=AU:en"
    else:
        # Default to US locale for international stocks
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(rss_url)

    rows = []
    for entry in getattr(feed, "entries", []):
        try:
            if hasattr(entry, "published_parsed"):
                d = datetime(*entry.published_parsed[:6])
                date_str = d.strftime("%Y-%m-%d %H:%M:%S")
            else:
                date_str = "N/A"
            orig_src = entry.get("source", {}).get("title", "N/A")
            provider = orig_src if orig_src in ("Yahoo Finance", "Business Today") else "Google News"
            rows.append({
                "date": date_str,
                "title": entry.title,
                "url": entry.link,
                "source": provider,
                "original_source": orig_src,
                "company": company_name,
                "ticker": ticker,
            })
        except Exception:
            continue
    df = _merge_and_dedupe([pd.DataFrame(rows)])
    return df


def _read_cache(ticker: str) -> pd.DataFrame:
    path = _cache_path_for(ticker)
    if not os.path.exists(path):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    # Ensure schema
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col != "original_source" else "N/A"
    return df[OUTPUT_COLUMNS]


def _write_cache(ticker: str, df: pd.DataFrame) -> None:
    path = _cache_path_for(ticker)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.sort_values("date").to_csv(path, index=False)


def _fetch_news_range(ticker: str, start_date: datetime, end_date: datetime, source: str = "all", verbose: bool = False) -> pd.DataFrame:
    if source == "all":
        return _merge_and_dedupe([
            _fetch_news_google(ticker, start_date, end_date),
            _fetch_news_yahoo(ticker, start_date, end_date),
            _fetch_news_businesstoday(ticker, start_date, end_date, verbose=verbose),
        ])
    if source == "google":
        return _fetch_news_google(ticker, start_date, end_date)
    if source == "yahoo":
        return _fetch_news_yahoo(ticker, start_date, end_date)
    if source == "businesstoday":
        return _fetch_news_businesstoday(ticker, start_date, end_date, verbose=verbose)
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def get_stock_news(ticker: str, start_date: datetime, end_date: datetime, verbose: bool = False, source: str = "all", include_social: bool = False) -> pd.DataFrame:
    """Fetch cached news for a date window; fill cache edges if needed; merge providers consistently."""
    cache_df = _read_cache(ticker)
    cached_start = cache_df["date"].min() if not cache_df.empty else None
    cached_end = cache_df["date"].max() if not cache_df.empty else None

    need_before = cached_start is None or start_date < cached_start
    need_after = cached_end is None or end_date > cached_end

    parts: List[pd.DataFrame] = []
    if need_before:
        fetch_end = (cached_start - timedelta(days=1)) if cached_start is not None else end_date
        if fetch_end >= start_date:
            parts.append(_fetch_news_range(ticker, start_date, fetch_end, source=source, verbose=verbose))
    if need_after:
        fetch_start = (cached_end + timedelta(days=1)) if cached_end is not None else start_date
        if end_date >= fetch_start:
            parts.append(_fetch_news_range(ticker, fetch_start, end_date, source=source, verbose=verbose))

    if parts:
        merged = _merge_and_dedupe(parts + ([cache_df] if not cache_df.empty else []))
        if not merged.empty:
            _write_cache(ticker, merged)
            cache_df = merged
        if verbose:
            total_new = sum(len(p) for p in parts if not p.empty)
            print(f"  Appended {total_new} new articles to cache for {ticker}")
    elif verbose:
        print(f"  Using cached news for {ticker}; range already covered")

    # Filter to window and (optionally) fetch all providers again to enrich
    window_df = cache_df[(cache_df["date"] >= start_date) & (cache_df["date"] <= end_date)].copy()
    if source == "all":
        enr = _fetch_news_range(ticker, start_date, end_date, source="all", verbose=verbose)
        window_df = _merge_and_dedupe([window_df, enr])
        if not window_df.empty:
            # Update cache with enriched data
            _write_cache(ticker, _merge_and_dedupe([cache_df, window_df]))

    # Source filter (exclusive) when not 'all'
    if source != "all" and not window_df.empty:
        if source == "yahoo":
            window_df = window_df[window_df["source"].str.contains("Yahoo", case=False, na=False)].copy()
        elif source == "google":
            window_df = window_df[window_df["source"].str.contains("Google News", case=False, na=False)].copy()
        elif source == "businesstoday":
            window_df = window_df[window_df["source"].str.contains("Business Today", case=False, na=False)].copy()
    
    # Include social media data (Reddit, Google Trends)
    if include_social:
        try:
            from sentiment.crawl_social import get_social_media_data
            from utils.color_log import social
            
            # Fetch social media data (cache check is handled upstream in sentiment_cache.py)
            if verbose:
                social("Including social media data (Reddit, Google Trends)...")
            social_df = get_social_media_data(ticker, start_date, end_date, sources=['reddit'], verbose=verbose)
            
            if not social_df.empty:
                # Merge new social posts with existing data (deduplicate by url)
                window_df = _merge_and_dedupe([window_df, social_df])
                
                # Update cache with new social media posts (save for next time)
                if not cache_df.empty:
                    updated_cache = _merge_and_dedupe([cache_df, social_df])
                    _write_cache(ticker, updated_cache)
                else:
                    # If no cache exists yet, create one with social posts
                    _write_cache(ticker, social_df)
                
                if verbose:
                    # Count new posts by comparing with existing cache
                    cached_social = cache_df[cache_df["source"].str.contains("Reddit|Google Trends", case=False, na=False)].copy() if not cache_df.empty else pd.DataFrame()
                    if cached_social.empty:
                        new_count = len(social_df)
                    else:
                        # Check which URLs are new
                        existing_urls = set(cached_social['url'].unique()) if 'url' in cached_social.columns else set()
                        new_count = len(social_df[~social_df['url'].isin(existing_urls)]) if 'url' in social_df.columns else len(social_df)
                    
                    if new_count > 0:
                        print(f"  Added {new_count} new social media posts")
                    else:
                        print(f"  All {len(social_df)} social media posts already cached")
        except Exception as e:
            if verbose:
                print(f"  WARNING: Could not fetch social media data: {e}")

    return window_df
