import os
import pickle
from loguru import logger

def ensure_directory_exists(directory):
    """
    Check if directory exists
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def save_data(data, file_path):
    """
    Save data to cache
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Data cached to {file_path}")

def create_scaler_cache_key(ticker: str, start_date: str, end_date: str,
                             target_feature: str, target_mode: str,
                             scale: bool, use_sentiment: bool) -> str:
    """
    Create cache key for scalers which depends on:
    - Ticker, dates (data source)
    - Target feature and mode (affects target scaler)
    - Scale mode (scaled vs raw)
    - Sentiment enabled (affects available columns)
    
    Args:
        ticker: Ticker symbol
        start_date: Start date
        end_date: End date
        target_feature: Target feature name
        target_mode: Target mode (price/return/log_return)
        scale: Whether scaling is enabled
        use_sentiment: Whether sentiment features are included
    
    Returns:
        Cache key string for scalers
    """
    # Normalize dates
    start_date = start_date.replace('/', '_').replace('\\', '_').replace('T', '_').split('_')[0]
    end_date = end_date.replace('/', '_').replace('\\', '_').replace('T', '_').split('_')[0]
    
    cache_key = (
        f"{ticker}_{start_date}_to_{end_date}_"
        f"{target_feature}_{target_mode}_"
        f"{'scaled' if scale else 'raw'}_{'sent' if use_sentiment else 'nosent'}"
    )
    return cache_key.replace('/', '_').replace('\\', '_')

def save_scalers(scalers: dict, scaler_cache_key: str, cache_dir='cache'):
    """
    Save scalers to cache for inference reuse.
    
    Args:
        scalers: Dict of scalers {feature_name: StandardScaler, '__target__': target_scaler}
        scaler_cache_key: Scaler cache key (from create_scaler_cache_key)
        cache_dir: Cache directory
    
    Note:
        Multiple feature subsets can reuse the same scalers dict by selecting relevant keys.
    """
    ensure_directory_exists(os.path.join(cache_dir, 'scalers'))
    scalers_path = os.path.join(cache_dir, 'scalers', f'{scaler_cache_key}_scalers.pkl')
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    logger.info(f"Scalers cached to {scalers_path}")
    return scalers_path

def load_scalers(scaler_cache_key: str, cache_dir='cache'):
    """
    Load cached scalers for inference.
    
    Args:
        scaler_cache_key: Scaler cache key (from create_scaler_cache_key)
        cache_dir: Cache directory
    
    Returns:
        Dict of scalers or None if not found
    """
    scalers_path = os.path.join(cache_dir, 'scalers', f'{scaler_cache_key}_scalers.pkl')
    if os.path.exists(scalers_path):
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        logger.info(f"Loaded cached scalers from {scalers_path}")
        return scalers
    else:
        logger.debug(f"No cached scalers found at {scalers_path}")
        return None

def check_scalers_exist(scaler_cache_key: str, cache_dir='cache'):
    """
    Check if cached scalers exist (without loading).
    
    Args:
        scaler_cache_key: Scaler cache key (from create_scaler_cache_key)
    """
    scalers_path = os.path.join(cache_dir, 'scalers', f'{scaler_cache_key}_scalers.pkl')
    return os.path.exists(scalers_path)

def save_processed_bundle(bundle, cache_key: str, cache_dir='cache'):
    """
    Save processed DataBundle to cache for reuse.
    
    Args:
        bundle: DataBundle object (with X_train, y_train, X_test, etc.)
        cache_key: Unique cache key (e.g., from create_bundle_cache_key)
        cache_dir: Cache directory
    """
    ensure_directory_exists(os.path.join(cache_dir, 'processed_data'))
    bundle_path = os.path.join(cache_dir, 'processed_data', f'{cache_key}_bundle.pkl')
    with open(bundle_path, 'wb') as f:
        pickle.dump(bundle, f)
    logger.info(f"Processed bundle cached to {bundle_path}")
    return bundle_path

def load_processed_bundle(cache_key: str, cache_dir='cache'):
    """
    Load cached processed DataBundle.
    
    Args:
        cache_key: Unique cache key
        cache_dir: Cache directory
    
    Returns:
        DataBundle or None if not found
    """
    bundle_path = os.path.join(cache_dir, 'processed_data', f'{cache_key}_bundle.pkl')
    if os.path.exists(bundle_path):
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)
        logger.info(f"Loaded cached bundle from {bundle_path}")
        return bundle
    else:
        logger.debug(f"No cached bundle found at {bundle_path}")
        return None

def create_bundle_cache_key(ticker: str, start_date: str, end_date: str, 
                            lookback: int, horizon: int, target_feature: str,
                            target_mode: str, scale: bool, use_sentiment: bool,
                            features: list) -> str:
    """
    Since we have multiple features options (sentiment, technical indicators, etc.), 
    We need a unique cache key for processed DataBundle to avoid data mismatches.
    We will use the sorted feature abbreviated names directly to create a compact representation or hash if too many features.
    
    Args:
        ticker: Ticker symbol
        start_date: Start date
        end_date: End date
        lookback: Lookback window size
        horizon: Prediction horizon
        target_feature: Target feature name
        target_mode: Target mode (price/return/log_return)
        scale: Whether scaling is enabled
        use_sentiment: Whether sentiment features are included
        features: List of feature names
    
    Returns:
        Cache key string
    """
    # Normalize dates
    start_date = start_date.replace('/', '_').replace('\\', '_').replace('T', '_').split('_')[0]
    end_date = end_date.replace('/', '_').replace('\\', '_').replace('T', '_').split('_')[0]
    
    # Use sorted feature names directly - create compact representation
    # For long feature lists, use abbreviated form to keep cache key manageable
    features_sorted = sorted(features)
    
    # If too many features, use count + hash (stable MD5) instead of full list
    if len(features_sorted) > 10:
        import hashlib
        features_str = ','.join(features_sorted)
        # Use MD5 for stable hash across sessions
        features_hash = hashlib.md5(features_str.encode()).hexdigest()[:8]
        features_repr = f"{len(features_sorted)}feat-{features_hash}"
    else:
        # For small feature lists, use feature names directly
        features_repr = '-'.join(features_sorted).replace(' ', '_')
    
    cache_key = (
        f"{ticker}_{start_date}_to_{end_date}_"
        f"L{lookback}_H{horizon}_{target_feature}_{target_mode}_"
        f"{'scaled' if scale else 'raw'}_{'sent' if use_sentiment else 'nosent'}_"
        f"feat[{features_repr}]"
    )
    return cache_key.replace('/', '_').replace('\\', '_')
