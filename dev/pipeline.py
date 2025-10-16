"""
Minimal ML pipeline wrapper.
Functions: prepare, train, predict, to_price, score.
Single source of truth for data contract and transforms.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Any
from sentiment.sentiment_pipeline import integrate_news_sentiment
from schemas.bundle import DataBundle, TargetMode
from dataio.converters import (
    clean_data, build_target, time_split, scale_features, scale_target, windows_train_val_test,
)
from dataio.postprocess import returns_to_prices, descale
from dataio.loading import load_stock_data


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
) -> DataBundle:
    """
    Prepare data as validated Pydantic DataBundle.

    Returns:
        DataBundle: Validated bundle with X:(N,L,F), y:(N,K)
    """
    # 1. Load
    df = load_stock_data(ticker, start_date, end_date, cache_dir=cache_dir)
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

    # 3.5 (optional): Integrate sentiment features before split
    if use_sentiment:
        print("  Integrating sentiment features...")
        df = integrate_news_sentiment(df, ticker, start_date, end_date)

    print("\nFull DataFrame (tail 5):")        
    print(df.tail(5))
    
    # 4. Split
    train_df, test_df, val_df = time_split(df, test_size=test_size, val_size=val_size)

    # 5. Scale (single source of truth)
    features = ['close', 'high', 'low', 'open', 'volume']
    # If sentiment columns exist, include them as additional features
    sent_extra = [c for c in ['s_mean', 's_median', 's_trim10', 'pos_ratio', 'neg_ratio', 'entropy'] if c in df.columns]
    if use_sentiment and sent_extra:
        features = features + sent_extra
    if scale:
        train_scaled, test_scaled, val_scaled, scalers = scale_features(
            train_df, test_df, features, val_df=val_df
        )
        print("\n[Preview] Train scaled features (head 3):")
        print(pd.DataFrame(train_scaled[:3], columns=features))
    else:
        train_scaled = train_df[features].values
        test_scaled = test_df[features].values
        val_scaled = val_df[features].values if val_df is not None else None
        scalers = {}

    # Extract target values using transformed column name
    target_train = train_df[target_col_name].to_numpy(dtype=float)
    target_test = test_df[target_col_name].to_numpy(dtype=float)
    target_val = val_df[target_col_name].to_numpy(dtype=float) if val_df is not None else None

    # Scale target if using scaling (consistent scaling for all modes)
    target_scaler = None
    if scale:
        target_train, target_test, target_val, target_scaler = scale_target(
            target_train, target_test, target_val
        )
        scalers['__target__'] = target_scaler

    # 6. Create windows with proper alignment (single source of truth)
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

    # 7. Base prices for test set (use original target_feature for return→price conversion)
    n_test_samples = X_test.shape[0]
    base_start = 0
    base_end = base_start + n_test_samples
    base_prices_test = test_df[target_feature].iloc[base_start:base_end].to_numpy()

    target_scaler = scalers.get("__target__", None)

    # 8. Build bundle
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

    print(bundle.summary())
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
