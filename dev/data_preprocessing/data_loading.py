import os
import pandas as pd
import yfinance as yf
from utils.file_handling import save_data


def load_stock_data(company, start_date, end_date, cache_dir='dev/cache/raw_data'):
    """
    Load stock data with caching to avoid repeated downloads.
    - Writes/reads a local pickle keyed by ticker and date range
    - Flattens yfinance MultiIndex columns (e.g., ('Close','')) for consistency
    """    
    # Compose a cache filename
    file_path = os.path.join(cache_dir,"raw_data", f"{company}_{start_date}_to_{end_date}.pkl")
    
    # Check if cached data exists
    if os.path.exists(file_path):
        print(f"Loading cached data for {company}")
        data = pd.read_pickle(file_path)
        # Also normalize cached data columns if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    else:
        print(f"Downloading data for {company}")
        data = yf.download(tickers=company, start=start_date, end=end_date)
        
        if data is not None and not data.empty:
            # yfinance may return a MultiIndex; flatten to a single level
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Save to cache
            save_data(data, file_path)
        else:
            raise ValueError(f"Failed to download data for {company}")
    return data