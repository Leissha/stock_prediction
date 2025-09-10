import os
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
        start_date=START_DATE,
        end_date=END_DATE,
        ticker=TICKER,
        lag_days=LAG_DAYS,
        lookup_step=LOOKUP_STEP,
        splitting_method=SPLIT_METHOD,
        test_size=TEST_SIZE,
        target_feature=TARGET_FEATURE,
        shuffle=SHUFFLE,
        scale=SCALE,
        target_as_return: bool = False,
        use_log_returns: bool = False,
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
            target_feature (str): Single target feature to predict (default: 'close')
            shuffle (bool): Whether to shuffle training sequences after creation
            scale (bool): Whether to scale features (Requirement e)
            base_path (str): Base filename for local caching (Requirement d)
            
        Returns:
            dict: Complete processed dataset with metadata
            
        """
        # Step 1: Load data with caching 
        df = load_stock_data(ticker, start_date, end_date, cache_dir=self.cache_dir)
            
        print(f"Loaded data: {df.shape} from {start_date} to {end_date}")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Standardize column names to lowercase for consistency
        df.columns = df.columns.str.lower()
        
        # Create chart for data inspection
        try:
            chart_path = f"cache/inspect_data/{ticker}_{start_date}_to_{end_date}"
            os.makedirs(os.path.dirname(chart_path), exist_ok=True)
            
            candlestick_path = f"{chart_path}/candlestick_chart.png"
            boxplot_path = f"{chart_path}/boxplot.png"
            
            # Check if chart files exist
            if not os.path.exists(candlestick_path):
                create_candlestick_chart(df, ticker, save_path=candlestick_path, n_days=1)  
                print(f"Candlestick chart saved to: {candlestick_path}")
            else:
                print("Candlestick chart already exists")
                
            if not os.path.exists(boxplot_path):
                create_boxplot(df, ticker, save_path=boxplot_path, n_days=20)
                print(f"Boxplot saved to: {boxplot_path}")
            else:
                print("Boxplot already exists")
                
        except Exception as e:
            logger.error(f"Chart creation failed: {e}")

        # Set default target feature if none specified
        if target_feature is None:
            target_feature = 'close'  # Default to close price
        else:
            target_feature = target_feature.lower()

        # Instead of price prediction, we can predict returns (simple % or log) for better model prediction
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pct_change.html 
        # e.g: OG y target [close]: 100, 105, 102, 103, 104, 105
        # Return y target [close_return]: 0, 0.05, -0.0286, 0.0097, 0.0096, 0.0095
        if target_as_return:
            target_feature = return_feature(df, target_feature, use_log_returns=use_log_returns)

        # Determine training features
        training_features = df.columns.tolist()

        print(f"Input features (OHLCV): {training_features}")
        print(f"Target feature: {target_feature}")
        
        # Step 2: Handle missing values 
        df = handle_nans(df)
        
        # Step 3: Split data 
        train_df, test_df = split_data(df, method=splitting_method, test_size=test_size, random_state=42)
        print(f"Data split - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Step 4: Feature scaling with separate scalers per feature 
        if scale:
            # Create separate scalers for each feature
            # Reason: OHLCV features have different scales and distributions:
            # - Price features (O,H,L,C): typically $10-$1000 range
            # - Volume feature: millions of shares (1,000,000+)
            # - Separate scaling preserves individual feature characteristics
            
            train_scaled = np.zeros_like(train_df[training_features].values)
            test_scaled = np.zeros_like(test_df[training_features].values)
            
            for i, feature in enumerate(training_features):
                # Create and store individual scaler for each feature
                scaler_key = f"{ticker}_{feature}"
                # self.scalers[scaler_key] = MinMaxScaler()
                self.scalers[scaler_key] = StandardScaler()
                
                # Fit scaler only on training data to prevent data leakage
                train_scaled[:, i] = self.scalers[scaler_key].fit_transform(
                    train_df[[feature]].values
                ).reshape(-1)
                test_scaled[:, i] = self.scalers[scaler_key].transform(
                    test_df[[feature]].values
                ).reshape(-1)
            # Cache all feature scalers for future inference use
            scaler_bundle = {
                "scalers": self.scalers,                 # per-feature scalers dict
                "training_features": training_features,      # column order
                "target_scaler_key": f"{ticker}_{target_feature}",  # just the key string
            }
            save_data(scaler_bundle, os.path.join(self.cache_dir, "scalers", f"{ticker}_{start_date}_to_{end_date}_scalers.pkl"))
            
            print(f"Applied separate StandardScaler to {len(training_features)} features")
        else:
            train_scaled = train_df[training_features].values
            test_scaled = test_df[training_features].values
            print("No scaling applied")
        
        # Step 5: Create batch sequences with future prediction
        print(f"Training:")
        X_train, y_train = create_sequences(
            train_scaled, lag_days, lookup_step, [target_feature], training_features
        )
        print(f"Testing:")
        test_scaled = np.vstack([train_scaled[-lag_days:], test_scaled])
        X_test, y_test = create_sequences(
            test_scaled, lag_days, lookup_step, [target_feature], training_features
        )
        
        # If target is a return column, remove it from X channels
        if target_feature.endswith('_return') and target_feature in training_features:
            t_idx = training_features.index(target_feature)
            X_train = np.delete(X_train, t_idx, axis=2)
            X_test = np.delete(X_test, t_idx, axis=2)
            training_features = [c for c in training_features if c != target_feature]
        
        # Step 6: Optional shuffling for training data
        if shuffle:
            # Only shuffle training data to maintain test set integrity
            indices = np.random.permutation(len(X_train))
            X_train, y_train = X_train[indices], y_train[indices]


        # Prepare comprehensive results dictionary
        result = {
            # Processed sequences for model training
            'X_train': X_train,           # Training input sequences
            'X_test': X_test,             # Test input sequences  
            'y_train': y_train,           # Training target values
            'y_test': y_test,             # Test target values
            'test_df': test_df,           # Test dataframe
            
            # Metadata for model configuration
            'training_features': training_features,    # Input feature names
            'target_feature': target_feature,      # Target feature name
            'scalers': self.scalers,               # Scalers dict
            # Save last available training-day price
            'last_training_price': train_df[target_feature.replace('_return','')].iloc[-1]
        }
        
        # Log final statistics
        print("=" * 60)
        print("DATA PROCESSING COMPLETE")
        print(f"Training sequences: X{X_train.shape} → y{y_train.shape}")
        print(f"Test sequences: X{X_test.shape} → y{y_test.shape}")
        
        if len(X_train) > 0:
            print(f"Data value range: [{X_train.min():.4f}, {X_train.max():.4f}]")
            
        print(f"Total scalers stored: {len(self.scalers)}")
        print("=" * 60)
        
        return result

def return_feature(df, target_feature, use_log_returns):
    """
    Original in/out features:
        training_features = ['close', 'high', 'low', 'open', 'volume']
        target_feature = 'close' 

    New in/out with return feature:
    # Note that close_return not in the training set to avoid data leakage 
        training_features = ['close', 'high', 'low', 'open', 'volume']     
        target_feature = 'close_return'
        
    OG y target     [close]       : 100,   105,    102,    103,    104,    105
    Return y target [close_return]:  0,   0.05, -0.0286, 0.0097, 0.0096, 0.0095
    """
    base_col = target_feature
    return_col = f"{base_col}_return"
    if use_log_returns:
        # Log returns: r_t = log(p_t) - log(p_{t-1})
        df[return_col] = df[base_col].astype(float).apply(np.log).diff().fillna(0).values
    else:
        # Simple percentage returns: r_t = (p_t - p_{t-1}) / p_{t-1}
        # More intuitive, but asymmetric (can't lose >100% but can gain unlimited)
        df[return_col] = df[base_col].pct_change().fillna(0).values

    # Set the new target feature to the engineered return column
    target_feature = return_col
        
    # Validate target feature exists in the dataframe
    if target_feature not in df.columns:
        raise ValueError(f"Target feature '{target_feature}' not found in df. Available columns: {list(df.columns)}")

    return target_feature