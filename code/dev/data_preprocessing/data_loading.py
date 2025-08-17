import os
import pandas as pd
from loguru import logger
import yfinance as yf
from utils.file_handling import ensure_directory_exists, check_file_existence, save_data


def load_stock_data(company, start_date, end_date, cache_dir='dev/cache/raw_data'):
    """
    Load stock data with caching to avoid repeated downloads
    Args:
        company: The company to load data for
        start_date: The start date to load data for
        end_date: The end date to load data for
        cache_dir: The directory to cache the data in
    Returns:
        data: The loaded data
    """
    # Create cache directory if it doesn't exist
    cache_dir = ensure_directory_exists(cache_dir)
    
    # Create cache filename
    file_path = os.path.join(cache_dir, f"{company}_{start_date}->{end_date}.pkl")
    
    # Check if cached data exists
    data = check_file_existence(file_path)
    if data is None:
        logger.info(f"Downloading data for {company}")
        data = yf.download(tickers=company, start=start_date, end=end_date)
        
        # Flatten MultiIndex columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Save to cache
        save_data(data, file_path)
    else:
        # Ensure cached data also has flattened columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
    
    return data