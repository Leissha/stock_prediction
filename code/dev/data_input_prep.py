import os
from loguru import logger
import numpy as np
from utils.file_handling import check_file_existence, save_data
from config.data import COMPANY, TRAIN_START, TRAIN_END
from data_preprocessing import train_test_split, normalize_data, create_sliding_window


def prepare_data_input(raw_data, company = COMPANY, price_value="Close", prediction_days=60, start_date=TRAIN_START, end_date=TRAIN_END, cache_dir='dev/cache/processed_data', load_data=True):
    """
    Prepare training data with caching and support for different price values
    Args:
        data: The raw data
        price_value: The price value to use for training 
        prediction_days: The number of days to predict
        cache_dir: The directory to store the prepared data
    Returns:
        scaler: The scaler used to normalize the data
        train_data: The training data (normalized and split)
        --> x_train: The features from train_data
        --> y_train: The labels from train_data
        test_data: The testing data (normalized and split)

    """ 
    # Create cache filename for prepared data
    filename = os.path.join(cache_dir, f"prepared_{company}_{start_date}->{end_date}.pkl")
    
    # Check if prepared data exists
    prepared_data = check_file_existence(filename)
    
    #------------------------------------------------------------------------------
    # 1. Load Data with caching
    #------------------------------------------------------------------------------

    # If prepared data exists, return the prepared data
    if prepared_data is not None:
        return {
            'scaler': prepared_data['scaler'],
            'x_train': prepared_data['x_train'],
            'y_train': prepared_data['y_train'],
            'x_test': prepared_data['x_test'],
            'y_test': prepared_data['y_test']
        }
    
    #------------------------------------------------------------------------------
    # TODO 2: Prepare Data with caching and different price values
    #------------------------------------------------------------------------------

    # Else, process the data
    logger.info("No prepared data found, processing data...")
    
    # Calculate price value can be (Open, High, Low, Volume, AdjClose)
    price_data = raw_data[price_value]
    
    #------------------------------------------------------------------------------
    # TODO 4: Better data processing for train/test split
    #------------------------------------------------------------------------------

    # Split the data
    train_data, test_data = train_test_split(price_data, test_ratio=0.2)
    
    # Normalize the data (returns 2D array)
    scaled_train_data, scaler = normalize_data(train_data)
    scaled_test_data = scaler.transform(test_data.values.reshape(-1, 1))  
    
    # Create sliding window (requires 1D array -> flatten the 2D array)
    x_train, y_train = create_sliding_window(scaled_train_data.reshape(-1), prediction_days)
    x_test, y_test = create_sliding_window(scaled_test_data.reshape(-1), prediction_days) 
    
    # Reshape for LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Cache the prepared data
    prepared_data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'scaler': scaler,
    }
    
    logger.info(f"Preview processed data: {prepared_data}")
    save_data(prepared_data, filename)
    
    return prepared_data