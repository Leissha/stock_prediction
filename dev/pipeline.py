"""
ML pipeline wrapper.
Functions: prepare, train, predict, to_price, score.
Orchestrates data contract and transforms.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Any
from sentiment.sentiment_cache import get_sentiment_cache
from schemas.bundle import DataBundle, TargetMode
from dataio.converters import (
    clean_data, build_target, time_split, scale_features, scale_target, windows_train_val_test,
)
from dataio.postprocess import returns_to_prices, descale
from dataio.loading import load_stock_data_by_ticker


def calculate_technical_indicators(df):
    """Calculate technical indicators for a given DataFrame."""
    if 'close' in df.columns:
        close = df['close']
        # SMA/EMA (20)
        df['sma_20'] = close.rolling(window=20, min_periods=20).mean()
        df['ema_20'] = close.ewm(span=20, adjust=False).mean()
        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        # MACD (12,26,9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = macd - signal
    return df


def prepare_data(
    ticker: str,
    start_date: str,
    end_date: str,
    target_feature: str = 'close',
    lookback: int = 60,
    horizon: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.0,
    target_as_return: bool = False,
    use_log_returns: bool = False,
    scale: bool = True,
    cache_dir: str = 'cache',
    use_sentiment: bool = False,
    include_social: bool = False,
    news_source: str = 'all',
) -> DataBundle:
    """
    Prepare data as validated Pydantic DataBundle.

    Returns:
        DataBundle: Validated bundle with X:(N,L,F), y:(N,K)
    """
    # Helper: Get target mode string for cache key
    from schemas.bundle import TargetMode
    target_mode_str = (TargetMode.LOG_RETURN.value if use_log_returns 
                      else TargetMode.RETURN.value if target_as_return 
                      else TargetMode.PRICE.value)
    
    # 1. Load (using ticker-based cache & date filtering for efficiency)
    df = load_stock_data_by_ticker(ticker, start_date=start_date, end_date=end_date, cache_dir=cache_dir)
    df.columns = df.columns.str.lower()

    # 2. Clean
    df = clean_data(df)

    # 3. Build target
    target_feature = target_feature.lower()
    if target_as_return:
        mode = "return"
        target_mode = TargetMode.LOG_RETURN if use_log_returns else TargetMode.RETURN
    else:
        mode = "price"
        target_mode = TargetMode.PRICE

    target_series, target_col_name = build_target(df, target_feature, mode=mode, use_log=use_log_returns)
    # target_feature = original column name (e.g., "close")
    # target_col_name = transformed column name (e.g., "close_return", "close_log_return")
    
    # Align df to target series index when returns/log drops first row
    if len(target_series) != len(df):
        df = df.loc[target_series.index].copy()
    if target_col_name not in df.columns:
        df[target_col_name] = target_series.astype(float).values

    # (optional): Integrate sentiment features before split
    if use_sentiment:
        from utils.color_log import sentiment
        source_desc = 'Google News + Yahoo Finance + Business Today'
        if include_social:
            source_desc += ' + Reddit + Google Trends'
        sentiment(f"Integrating sentiment features (source: {source_desc if news_source == 'all' else news_source})...")
        sentiment_cache = get_sentiment_cache()
        df = sentiment_cache.integrate_sentiment(df, ticker, start_date, end_date, source=news_source, include_social=include_social)

    print("\nFull DataFrame (tail 5):")
    print(df.tail(5))
    
    # 4. Split
    train_df, test_df, val_df = time_split(df, test_size=test_size, val_size=val_size)

    # 5. Calculate technical indicators separately for each split
    train_df = calculate_technical_indicators(train_df)
    test_df = calculate_technical_indicators(test_df)
    if val_df is not None:
        val_df = calculate_technical_indicators(val_df)

    # 6. Scale
    features = ['close', 'high', 'low', 'open', 'volume']
    tech_cols = [c for c in ['sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'macd_hist'] if c in train_df.columns]
    if tech_cols:
        features = features + tech_cols
    # If sentiment columns exist, include ALL sentiment features dynamically
    if use_sentiment:
        # Dynamically detect sentiment columns (exclude base OHLCV/target/dates and technical indicators)
        exclude_cols = set(['close','high','low','open','volume', target_col_name, 'date_only', 'date'] + tech_cols)
        sentiment_features = [c for c in df.columns if c not in exclude_cols]
        if sentiment_features:
            features = features + sentiment_features
            print(f"  Using {len(sentiment_features)} sentiment features: {sentiment_features[:5]}..." +
                  (f" and {len(sentiment_features)-5} more" if len(sentiment_features) > 5 else ""))
    
    # Check for cached processed bundle now that we have features list
    from utils.file_handling import create_bundle_cache_key, load_processed_bundle, save_processed_bundle, save_scalers
    cache_key = create_bundle_cache_key(
        ticker, start_date, end_date, lookback, horizon,
        target_feature, target_mode_str, scale, use_sentiment, features
    )
    cached_bundle = load_processed_bundle(cache_key, cache_dir=cache_dir)
    if cached_bundle is not None:
        print(f"Loaded cached processed bundle (skip pipeline processing)")
        return cached_bundle
    if scale:
        train_scaled, test_scaled, val_scaled, scalers = scale_features(
            train_df, test_df, features, val_df=val_df
        )
    else:
        train_scaled = train_df[features].values
        test_scaled = test_df[features].values
        val_scaled = val_df[features].values if val_df is not None else None
        scalers = {}

    # Handle NaNs after scaling to prevent network collapse as technical indicators like sma_20 has NaNs for the first few rows for rolling calculations
    train_scaled = np.nan_to_num(train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    test_scaled = np.nan_to_num(test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    if val_scaled is not None:
        val_scaled = np.nan_to_num(val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    print("\n[Preview] Train scaled features (head 3):")
    print(pd.DataFrame(train_scaled[:3], columns=features))

    # Extract target values using transformed column name
    target_train = train_df[target_col_name].to_numpy(dtype=float)
    target_test = test_df[target_col_name].to_numpy(dtype=float)
    target_val = val_df[target_col_name].to_numpy(dtype=float) if val_df is not None else None

    # Scale target if using scaling
    target_scaler = None
    if scale:
        target_train, target_test, target_val, target_scaler = scale_target(
            target_train, target_test, target_val
        )
        scalers['__target__'] = target_scaler

    # Ensure target data is reshaped for binary classification
    if target_mode == TargetMode.PRICE and target_train.ndim > 1:
        target_train = target_train[:, 0:1]
        target_test = target_test[:, 0:1]
        if target_val is not None:
            target_val = target_val[:, 0:1]

    # 7. Create windows with proper alignment
    X_train, y_train, X_test, y_test, X_val, y_val = windows_train_val_test(
        train_scaled, test_scaled, target_train, target_test,
        lookback=lookback, horizon=horizon,
        val_scaled=val_scaled, target_val=target_val
    )

    # Calculate base prices for validation (if provided)
    # Use original target_feature for base prices (needed for return→price conversion)
    if X_val is not None and val_df is not None:
        n_val_samples = X_val.shape[0]
        base_prices_val = val_df[target_feature].iloc[:n_val_samples].to_numpy()
    else:
        base_prices_val = None

    # 8. Base prices for test set (use original target_feature for return→price conversion)
    n_test_samples = X_test.shape[0]
    base_start = 0
    base_end = base_start + n_test_samples
    base_prices_test = test_df[target_feature].iloc[base_start:base_end].to_numpy()

    target_scaler = scalers.get("__target__", None)

    # 9. Build bundle
    bundle = DataBundle(
        symbol=ticker,
        features=features,
        lookback=lookback,
        horizon=horizon,
        target_mode=target_mode,
        use_log_returns=use_log_returns,
        use_sentiment=use_sentiment,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        scalers=scalers,
        base_prices_test=base_prices_test,
        base_prices_val=base_prices_val,
        target_scaler=target_scaler,
        # Store original DataFrames for SARIMAX
        train_df=train_df.copy(),
        test_df=test_df.copy(),
        val_df=val_df.copy() if val_df is not None else None,
        target_feature=target_col_name
    )
    
    # 10. Cache processed bundle for reuse
    # Use cache_key already created above (line 133)
    save_processed_bundle(bundle, cache_key, cache_dir=cache_dir)
    print(f"Processed bundle cached: {cache_key[:80]}...")
    
    # 11. Cache scalers using scaler cache key
    if scale and scalers:
        from utils.file_handling import create_scaler_cache_key
        scaler_cache_key = create_scaler_cache_key(
            ticker, start_date, end_date,
            target_feature, target_mode_str,
            scale, use_sentiment
        )
        save_scalers(scalers, scaler_cache_key, cache_dir=cache_dir)
        print(f"Scalers cached: {scaler_cache_key[:80]}...")
    
    return bundle


def predict(model: Any, bundle: DataBundle) -> np.ndarray:
    """
    Generate predictions.

    Returns:
        np.ndarray: Predictions shape (N_test, K)
    """
    if hasattr(model, 'predict'):
        # Generic sklearn-like
        try:
            preds = model.predict(bundle.X_test)
        except TypeError:
            # Some models expect both X and y; pass y for compatibility, but ignore metrics here
            preds = model.predict(bundle.X_test, bundle.y_test)  # type: ignore
        # Ensure 2D
        if preds.ndim == 1:
            preds = preds[:, None]
        return preds
    # No legacy predict_and_evaluate path; models must implement predict()
    else:
        raise ValueError(f"Model {type(model)} has no predict method")


def predictions_to_prices(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    bundle: DataBundle
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert returns to prices using per-row base prices.

    Returns:
        (y_true_px, y_hat_px): Both (N_test, K) in price space
    """
    # Descale targets if they were scaled
    if getattr(bundle, 'target_scaler', None) is not None:
        y_true = descale(y_true, bundle.target_scaler)
        y_hat = descale(y_hat, bundle.target_scaler)

    if bundle.target_mode == TargetMode.PRICE:
        # Already in price space after descaling
        return y_true, y_hat

    # Convert returns to prices (only for return/log_return)
    mode_value = bundle.target_mode.value if hasattr(bundle.target_mode, 'value') else bundle.target_mode

    assert bundle.base_prices_test is not None, "base_prices_test missing in bundle"
    y_true_px = returns_to_prices(
        y_true,
        bundle.base_prices_test,
        mode=mode_value,
        use_log=bundle.use_log_returns
    )

    y_hat_px = returns_to_prices(
        y_hat,
        bundle.base_prices_test,
        mode=mode_value,
        use_log=bundle.use_log_returns
    )

    # Validate no NaN/Inf/negative prices
    assert np.isfinite(y_true_px).all(), "y_true has non-finite values"
    assert np.isfinite(y_hat_px).all(), "y_hat has non-finite values"
    assert (y_true_px > 0).all(), f"y_true has non-positive prices: min={y_true_px.min()}"
    assert (y_hat_px > 0).all(), f"y_hat has non-positive prices: min={y_hat_px.min()}"

    return y_true_px, y_hat_px
