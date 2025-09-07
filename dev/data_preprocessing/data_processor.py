import os
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from config.data import *

# Import existing modular code
from .handle_nans import handle_nans
from .create_sequence import create_sequences
from .data_splitting import split_data
from .data_loading import load_stock_data
from utils.file_handling import save_data
from utils.plots import create_boxplot, create_candlestick_chart

class DataProcessor:
    """
    Processing pipeline:
    1. Load data with local caching & create chart for data inspection
    2. Handle missing values 
    3. Split data chronologically/randomly 
    4. Scale features with scaler storage 
    5. Create LSTM sequences for single-target prediction
    6. Optional shuffling for training data
    
    Additional features:
    - Flexible sequence creation for LSTM
    - Future prediction with lookup_step
    """
    
    def __init__(self, cache_dir='cache'):
        """
        Initialize the DataProcessor.
        
        Args:
            cache_dir (str): Directory for caching processed data locally
        """
        self.cache_dir = cache_dir
        self.scalers = {}  # Store scalers for future access 
        
        # Create cache directory if it doesn't exist 
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def data_processing(
        self,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        ticker=COMPANY,
        lag_days=LAG_DAYS,
        lookup_step=LOOKUP_STEP,
        splitting_method=SPLIT_METHOD,
        test_size=TEST_SIZE,
        target_feature=PRICE_VALUE,
        shuffle=SHUFFLE,
        scale=SCALE,
    ):
        """
        Main data processing function for multi-input, single-target stock prediction.
        
        Args:
            start_date (str): Start date for dataset (Requirement a)
            end_date (str): End date for dataset (Requirement a)
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
            lag_days (int): Historical sequence length for LSTM input
            lookup_step (int): Future prediction step (1=next day, 5=next week)
            splitting_method (str): Data splitting method - 'date', 'ratio', 'random' (Requirement c)
            test_size (float): Proportion of data for testing (0.0 to 1.0)
            feature_columns (list): Input features (OHLCV) for prediction
            target_feature (str): Single target feature to predict (default: 'close')
            shuffle (bool): Whether to shuffle training sequences after creation
            scale (bool): Whether to scale features (Requirement e)
            base_path (str): Base filename for local caching (Requirement d)
            
        Returns:
            dict: Complete processed dataset with metadata
            
        """
        # Step 1: Load data with caching 
        df = load_stock_data(ticker, start_date, end_date, cache_dir=self.cache_dir)
            
        logger.info(f"Loaded data: {df.shape} from {start_date} to {end_date}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Standardize column names to lowercase for consistency
        df.columns = df.columns.str.lower()
        feature_columns = df.columns.tolist()
        
        # Create chart for data inspection
        try:
            chart_path = f"cache/inspect_data/{ticker}_{start_date}_to_{end_date}"
            os.makedirs(os.path.dirname(chart_path), exist_ok=True)
            
            candlestick_path = f"{chart_path}/candlestick_chart.png"
            boxplot_path = f"{chart_path}/boxplot.png"
            
            # Check if chart files exist
            if not os.path.exists(candlestick_path):
                create_candlestick_chart(df, ticker, save_path=candlestick_path, n_days=1)  
                logger.info(f"Candlestick chart saved to: {candlestick_path}")
            else:
                logger.info("Candlestick chart already exists")
                
            if not os.path.exists(boxplot_path):
                create_boxplot(df, ticker, save_path=boxplot_path, n_days=20)
                logger.info(f"Boxplot saved to: {boxplot_path}")
            else:
                logger.info("Boxplot already exists")
                
        except Exception as e:
            logger.error(f"Chart creation failed: {e}")

        # Set default target feature if none specified
        if target_feature is None:
            target_feature = 'close'  # Default to close price
        else:
            target_feature = target_feature.lower()
            
        # Validate target feature is in feature columns
        if target_feature not in feature_columns:
            raise ValueError(f"Target feature '{target_feature}' must be in feature_columns: {feature_columns}")

        logger.info(f"Input features (OHLCV): {feature_columns}")
        logger.info(f"Target feature: {target_feature}")
        
        # Step 2: Handle missing values 
        df = handle_nans(df)
        
        # Step 3: Split data 
        train_df, test_df = split_data(df, method=splitting_method, test_size=test_size, random_state=42)
        logger.info(f"Data split - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Step 4: Feature scaling with separate scalers per feature 
        if scale:
            # Create separate scalers for each feature
            # Reason: OHLCV features have different scales and distributions:
            # - Price features (O,H,L,C): typically $10-$1000 range
            # - Volume feature: millions of shares (1,000,000+)
            # - Separate scaling preserves individual feature characteristics
            
            train_scaled = np.zeros_like(train_df[feature_columns].values)
            test_scaled = np.zeros_like(test_df[feature_columns].values)
            
            for i, feature in enumerate(feature_columns):
                # Create and store individual scaler for each feature
                scaler_key = f"{ticker}_{feature}"
                self.scalers[scaler_key] = MinMaxScaler()
                
                # Fit scaler only on training data to prevent data leakage
                train_scaled[:, i] = self.scalers[scaler_key].fit_transform(
                    train_df[[feature]].values
                ).reshape(-1)
                test_scaled[:, i] = self.scalers[scaler_key].transform(
                    test_df[[feature]].values
                ).reshape(-1)
                
                logger.info(f"Feature '{feature}' scaled: range {self.scalers[scaler_key].data_min_[0]:.4f} to {self.scalers[scaler_key].data_max_[0]:.4f}")
                
            # Cache all feature scalers for future inference use
            scaler_bundle = {
                "scalers": self.scalers,                 # per-feature scalers dict
                "feature_columns": feature_columns,      # column order
                "target_scaler_key": f"{ticker}_{target_feature}",  # just the key string
            }
            save_data(scaler_bundle, os.path.join(self.cache_dir, "scalers", f"{ticker}_{start_date}_to_{end_date}_scalers.pkl"))
            
            logger.info(f"Applied separate MinMaxScaler to {len(feature_columns)} features")
        else:
            train_scaled = train_df[feature_columns].values
            test_scaled = test_df[feature_columns].values
            logger.info("No scaling applied")
        
        # Step 5: Create batch sequences with future prediction
        logger.info(f"Creating sequences with lag_days={lag_days}, lookup_step={lookup_step}")
        
        X_train, y_train = create_sequences(
            train_scaled, lag_days, lookup_step, [target_feature], feature_columns
        )
        test_scaled = np.vstack([train_scaled[-lag_days:], test_scaled])
        X_test, y_test = create_sequences(
            test_scaled, lag_days, lookup_step, [target_feature], feature_columns
        )
        
        # Use existing feature scaler as target scaler (no duplication needed)
        if scale:
            # The feature scaler for target is already our target scaler
            target_key = f"{ticker}_{target_feature}"
            if target_key in self.scalers:
                # Copy the existing feature scaler as dedicated target scaler
                self.scalers[f"{ticker}_{target_feature}_target"] = self.scalers[target_key]
                logger.info(f"Using existing feature scaler as target scaler: range {self.scalers[target_key].data_min_[0]:.4f} to {self.scalers[target_key].data_max_[0]:.4f}")
            else:
                logger.warning(f"Target feature scaler not found: {target_key}")
        else:
            logger.info("No target scaler needed (scaling disabled)")
        
        # Extract current prices (last price in each input window) for trading metrics
        # These are the prices at time t for each sample (before prediction horizon)
        current_prices_train = train_df[target_feature].iloc[lag_days-1:-lookup_step].values

        # For test: derive sequence-aligned current prices from the last timestep of each test window
        # If scaled, inverse-transform using the target scaler; otherwise take values as-is.
        close_idx = feature_columns.index(target_feature)
        if scale:
            curr_scaled_test = X_test[:, -1, close_idx].reshape(-1, 1)
            current_prices_test = self.scalers[f"{ticker}_{target_feature}"].inverse_transform(curr_scaled_test).reshape(-1)
        else:
            current_prices_test = X_test[:, -1, close_idx].reshape(-1)

        # Sanity checks to ensure alignment
        assert len(y_train) == len(X_train) == len(current_prices_train), (
            f"Train length mismatch: X={len(X_train)}, y={len(y_train)}, curr={len(current_prices_train)}"
        )
        assert len(y_test) == len(X_test) == len(current_prices_test), (
            f"Test length mismatch: X={len(X_test)}, y={len(y_test)}, curr={len(current_prices_test)}"
        )
        
        # Step 6: Optional shuffling for training data
        if shuffle:
            # Only shuffle training data to maintain test set integrity
            indices = np.random.permutation(len(X_train))
            X_train, y_train = X_train[indices], y_train[indices]
            current_prices_train = current_prices_train[indices]  # Shuffle current prices too
            logger.info("Training sequences shuffled")


        # Prepare comprehensive results dictionary
        result = {
            # Processed sequences for model training
            'X_train': X_train,           # Training input sequences
            'X_test': X_test,             # Test input sequences  
            'y_train': y_train,           # Training target values
            'y_test': y_test,             # Test target values
            'current_prices_train': current_prices_train,  # Current prices for training
            'current_prices_test': current_prices_test,    # Current prices for testing
            'test_df': test_df,           # Test dataframe
            
            # Metadata for model configuration
            'feature_columns': feature_columns,    # Input feature names
            'target_feature': target_feature,      # Target feature name
            'scalers': self.scalers,               # Scalers dict
        }
        
        # Log final statistics
        logger.info("=" * 60)
        logger.info("DATA PROCESSING COMPLETE")
        logger.info(f"Training sequences: X{X_train.shape} → y{y_train.shape}")
        logger.info(f"Test sequences: X{X_test.shape} → y{y_test.shape}")
        
        if len(X_train) > 0:
            logger.info(f"Data value range: [{X_train.min():.4f}, {X_train.max():.4f}]")
            
        logger.info(f"Total scalers stored: {len(self.scalers)}")
        logger.info("=" * 60)
        
        return result

