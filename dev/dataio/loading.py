import os
import pandas as pd
import yfinance as yf
from utils.file_handling import save_data
from utils.color_log import cache, data as log_data


def load_stock_data(company, start_date, end_date, cache_dir='cache/raw_data'):
    """
    Load stock data with caching to avoid repeated downloads.
    - Writes/reads a local pickle keyed by ticker and date range
    - Flattens yfinance MultiIndex columns (e.g., ('Close','')) for consistency
    """
    # Compose a cache filename
    file_path = os.path.join(cache_dir, "raw_data", f"{company}_{start_date}_to_{end_date}.pkl")
    
    # Check if cached data exists
    if os.path.exists(file_path):
        cache(f"Loading cached data for {company}")
        df_data = pd.read_pickle(file_path)
        # Also normalize cached data columns if needed
        if isinstance(df_data.columns, pd.MultiIndex):
            df_data.columns = df_data.columns.get_level_values(0)
        return df_data
    else:
        log_data(f"Downloading data for {company}")
        df_data = yf.download(tickers=company, start=start_date, end=end_date)
        
        if df_data is not None and not df_data.empty:
            # yfinance may return a MultiIndex; flatten to a single level
            if isinstance(df_data.columns, pd.MultiIndex):
                df_data.columns = df_data.columns.get_level_values(0)
            
            # Save to cache
            save_data(df_data, file_path)
        else:
            raise ValueError(f"Failed to download data for {company}")
    return df_data


def load_stock_data_by_ticker(company, start_date=None, end_date=None, cache_dir='cache/raw_data', max_years=5):
    """
    Load stock data cached by ticker name (not date range).
    More efficient for dashboards - cache full history, filter by date in-memory.
    
    Args:
        company: Ticker symbol (e.g., 'AAPL')
        start_date: Optional start date filter (YYYY-MM-DD). If None, uses max_years ago.
        end_date: Optional end date filter (YYYY-MM-DD). If None, uses today.
        cache_dir: Cache directory
        max_years: Max years to download if cache doesn't exist (default 5)
    
    Returns:
        DataFrame filtered by date range (if provided)
    
    Benefits:
        - One cache file per ticker (e.g., 'AAPL.pkl')
        - Fast date filtering using pandas
        - Better for dashboards showing multiple stocks
    """
    import datetime as dt
    
    # Cache file: one per ticker
    cache_file = os.path.join(cache_dir, "raw_data", f"{company}.pkl")
    
    # Normalize date strings (remove time component if present)
    if isinstance(end_date, str) and 'T' in end_date:
        end_date = end_date.split('T')[0]
    if isinstance(start_date, str) and 'T' in start_date:
        start_date = start_date.split('T')[0]
    
    # Default date range if not provided
    if end_date is None:
        end_date = dt.date.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (dt.date.today() - dt.timedelta(days=max_years*365)).strftime("%Y-%m-%d")
    
    # Convert to datetime for filtering
    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()
    
    # Check if cached full history exists
    if os.path.exists(cache_file):
        cache(f"Loading cached data for {company} (filtering {start_date} to {end_date})")
        data = pd.read_pickle(cache_file)
        
        # Normalize columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Ensure index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Filter by date range (date search in pickle file)
        data = data[(data.index.date >= start_dt) & (data.index.date <= end_dt)]
        
        if data.empty:
            from utils.color_log import warning
            warning(f"No data for {company} in range {start_date} to {end_date}")
        
        return data
    else:
        # Download full history (up to max_years)
        log_data(f"Downloading data for {company} (last {max_years} years)")
        download_start = (dt.date.today() - dt.timedelta(days=max_years*365)).strftime("%Y-%m-%d")
        # Normalize end_date to date string (remove time component if present)
        if isinstance(end_date, str) and 'T' in end_date:
            end_date = end_date.split('T')[0]
        df_data = yf.download(tickers=company, start=download_start, end=end_date, progress=False, auto_adjust=True)
        
        if df_data is not None and not df_data.empty:
            # Normalize columns
            if isinstance(df_data.columns, pd.MultiIndex):
                df_data.columns = df_data.columns.get_level_values(0)
            
            # Save full history to cache (by ticker only)
            save_data(df_data, cache_file)
            cache(f"Cached full history for {company} ({len(df_data)} rows)")
            
            # Filter by requested date range
            if isinstance(df_data.index, pd.DatetimeIndex):
                df_data = df_data[(df_data.index.date >= start_dt) & (df_data.index.date <= end_dt)]
            
            return df_data
        else:
            raise ValueError(f"Failed to download data for {company}")