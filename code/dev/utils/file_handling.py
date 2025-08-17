import os
import pickle
from loguru import logger
import tensorflow as tf

def ensure_directory_exists(directory):
    """
    Check if directory exists
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def check_file_existence(file_path):
    """
    Check if cached data exists
    """
    if os.path.exists(file_path):
        logger.info(f"Loading cached data")
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        elif file_path.endswith('.h5'):
            return tf.keras.models.load_model(file_path)
    return None

def save_data(data, file_path):
    """
    Save data to cache
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Data cached to {file_path}")
