"""
Data Preprocessing Package for Stock Prediction

This package provides comprehensive data preprocessing functionality for stock prediction models.
It handles data loading, cleaning, splitting, scaling, and sequence creation for LSTM models.

Key Components:
- DataProcessor: Main class for end-to-end data processing
- Data loading with caching
- NaN handling
- Data splitting (chronological, ratio, random)
- Feature scaling
- Sequence creation for LSTM models

Usage:
    from dev.data_preprocessing import DataProcessor
    
    processor = DataProcessor(cache_dir='cache')
    data = processor.data_processing(
        start_date='2023-01-01',
        end_date='2023-12-31',
        ticker='AAPL',
        feature_columns=['close', 'open', 'high', 'low', 'volume'],
        target_feature='close'
    )
"""

from .data_processor import DataProcessor
from .data_loading import load_stock_data
from .handle_nans import handle_nans
from .create_sequence import create_sequences
from .data_splitting import split_data

__all__ = [
    'DataProcessor',
    'load_stock_data', 
    'handle_nans',
    'create_sequences',
    'split_data'
]