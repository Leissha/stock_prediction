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

def init_cache_dir():
    for dir in ["cache", "cache/trained_models", "cache/processed_data", "cache/raw_data", "cache/scalers", "results"]:
        os.makedirs(f"{dir}", exist_ok=True)

def save_data(data, file_path):
    """
    Save data to cache
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Data cached to {file_path}")

def load_scalers(ticker, cache_dir='cache'):
    """
    Load cached scalers for inference
    """
    scalers_path = os.path.join(cache_dir, 'scalers', f'{ticker}_scalers.pkl')
    return os.path.exists(scalers_path)
