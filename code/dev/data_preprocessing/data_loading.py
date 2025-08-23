import pandas as pd
from loguru import logger
import yfinance as yf
from utils.file_handling import check_file_existence, save_data


def load_stock_data(company, start_date, end_date, file_path):
    """
    Load stock data with caching to avoid repeated downloads
    Args:
        company: The company to load data for
        start_date: The start date to load data for
        end_date: The end date to load data for
        file_path: The path to the file to cache the data in
    Returns:
        data: The loaded data
    """
    # Check if cached data exists
    data = check_file_existence(file_path)
    if data is None:
        logger.info(f"Downloading data for {company}")
        data = yf.download(tickers=company, start=start_date, end=end_date)
        
        if data is not None and not data.empty:
            # Flatten MultiIndex columns if they exist
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Save to cache
            save_data(data, file_path)
        else:
            raise ValueError(f"Failed to download data for {company}")
    else:
        # Ensure cached data also has flattened columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
    
    return data