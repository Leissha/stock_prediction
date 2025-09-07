from loguru import logger
from sklearn.model_selection import train_test_split
from config.data import *

def split_data(df, method=SPLIT_METHOD, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split data using different methods (Requirement c)
    
    Args:
        df (pd.DataFrame): Input dataframe to split
        method (str): Splitting method - 'date', 'ratio', or 'random'
        test_size (float): Proportion of data for testing (0.0 to 1.0)
        random_state (int): Seed for reproducible random splits
        
    Returns:
        tuple: (train_df, test_df)
        
    Method explanations:
    - 'date': Chronological split, respects temporal order (recommended for time series)
    - 'ratio': Same as date but more explicit naming
    - 'random': Random sampling (breaks temporal dependencies, use with caution)
    """
    logger.info(f"Splitting data using method: {method}, test_size: {test_size}")
    
    if method in ['date', 'ratio']:
        # Chronological split - maintains temporal order
        # This is critical for time series to avoid data leakage
        split_point = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_point].copy()
        test_df = df.iloc[split_point:].copy()
        logger.info(f"Chronological split at index {split_point}")
        
    elif method == 'random':
        # Random split - WARNING: breaks temporal dependencies
        # Only use this for experimental purposes or when temporal order doesn't matter
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        logger.warning("Random split used - temporal dependencies broken!")
        
    else:
        raise ValueError(f"Unknown split method: {method}. Use 'date', 'ratio', or 'random'")
        
    return train_df, test_df



def val_split_with_prices(x_train, y_train, current_prices_train, val_size=0.2):
    """
    Split data into training and validation sets with current prices.
    """
    val_size = int(len(x_train) * val_size)
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    current_prices_val = current_prices_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]
    current_prices_train = current_prices_train[:-val_size]
    return x_train, y_train, x_val, y_val, current_prices_train, current_prices_val