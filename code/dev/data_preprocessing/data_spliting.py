from loguru import logger
import numpy as np


def train_test_split(data, test_ratio=0.2):
    """
    Simple train-test split method
    """
    # Calculate split point
    split_point = int(len(data) * (1 - test_ratio))
    
    # Split the data
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")
    
    return train_data, test_data
