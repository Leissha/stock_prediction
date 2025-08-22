from loguru import logger

from utils.data_handling import shuffle_in_unison
from sklearn.model_selection import train_test_split


def split_data(df, X, y, method='date', test_size=0.2, shuffle=True):
    """
    Split data using different methods.
    
    Args:
        data (pd.DataFrame): Data to split
        method (str): 'date', 'ratio', or 'random'
        test_size (float): Ratio for test data
        shuffle (bool): Whether to shuffle data
        
    Returns:
        tuple: (train_data, test_data)
    """
    logger.info(f"Splitting data using method: {method}")
    
    if method == 'date':
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        df["X_train"] = X[:train_samples]
        df["y_train"] = y[:train_samples]
        df["X_test"]  = X[train_samples:]
        df["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(df["X_train"], df["y_train"])
            shuffle_in_unison(df["X_test"], df["y_test"])
        
    elif method == 'ratio':
        # Split by ratio but maintain chronological order
        split_point = int(len(X) * (1 - test_size))
        df["X_train"] = X[:split_point]
        df["y_train"] = y[:split_point]
        df["X_test"] = X[split_point:]
        df["y_test"] = y[split_point:]
        if shuffle:
            shuffle_in_unison(df["X_train"], df["y_train"])
            shuffle_in_unison(df["X_test"], df["y_test"])
            
    elif method == 'random':
        # split the dataset randomly
        df["X_train"], df["X_test"], df["y_train"], df["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)
    else:
        raise ValueError(f"Unknown split method: {method}")
        
    return df