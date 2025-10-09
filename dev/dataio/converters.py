"""
Data converters: clean → target → split → scale → window.

All functions follow strict shape contracts:
- X: (N, L, F)
- y: (N, K)
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Handle missing values in OHLCV data.

    Args:
        df: DataFrame with OHLCV columns
        method: "ffill" (forward-fill) or "drop"

    Returns:
        Cleaned DataFrame

    Shape:
        Input: (M, C)
        Output: (M', C) where M' <= M
    """
    if method == "ffill":
        return df.ffill().dropna()
    elif method == "drop":
        return df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")


def build_target(
    df: pd.DataFrame, target_col: str, mode: str, use_log: bool = False
) -> Tuple[pd.Series, str]:
    """
    Create target column based on mode.

    Args:
        df: DataFrame with price columns
        target_col: Base column name (e.g., "close")
        mode: "price" | "return" | "log_return"
        use_log: If True and mode="return", use log returns

    Returns:
        (df_with_target, target_column_name)

    Shape:
        Input: (M, C)
        Output: (M-1, C+1) if mode="return", else (M, C) or (M, C+1)

    Formulas:
        - price: target = df[col]
        - return: r_t = (p_t - p_{t-1}) / p_{t-1} = pct_change()
        - log_return: r_t = log(p_t / p_{t-1})
    """
    target_col = target_col.lower()

    if mode == "price":
        # No transformation needed, return original price series
        return df[target_col].astype(float), target_col

    elif mode == "return":
        if use_log:
            # Log returns via diff of log price (keep as pandas Series)
            return_col = f"{target_col}_log_return"
            series = df[target_col].astype(float).apply(np.log).diff()
        else:
            # Simple percentage return (pandas Series)
            return_col = f"{target_col}_return"
            series = df[target_col].astype(float).pct_change()

        # Drop first row with NaN
        series = series.iloc[1:].astype(float)
        return series, return_col

    elif mode == "log_return":
        # Alias for return with use_log=True
        return_col = f"{target_col}_log_return"
        series = df[target_col].astype(float).apply(np.log).diff().iloc[1:].astype(float)
        return series, return_col

    else:
        raise ValueError(f"Unknown mode: {mode}")


def time_split(
    df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Chronological train/val/test split.

    Args:
        df: Time-indexed DataFrame
        test_size: Proportion for test (0.0-1.0)
        val_size: Proportion for validation (0.0-1.0)

    Returns:
        (train_df, test_df, val_df) - val_df is None if val_size=0

    Shape:
        Input: (M, C)
        Output: train:(T,C), test:(Te,C), val:(V,C) where T+Te+V=M
    """
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - val_size)) if val_size > 0 else test_idx

    # Ensure pandas DataFrame usage (avoid NDArray in type checker)
    train_df = pd.DataFrame(df).iloc[:val_idx].copy()
    val_df = pd.DataFrame(df).iloc[val_idx:test_idx].copy() if val_size > 0 else None
    test_df = pd.DataFrame(df).iloc[test_idx:].copy()

    return train_df, test_df, val_df


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    val_df: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, StandardScaler]]:
    """
    Fit scalers on train, transform all splits.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names to scale
        val_df: Optional validation DataFrame

    Returns:
        (train_scaled, test_scaled, val_scaled, scalers_dict)
        val_scaled is None if val_df is None

    Shape:
        train_df: (T, C) → train_scaled: (T, F)
        test_df: (Te, C) → test_scaled: (Te, F)
        val_df: (V, C) → val_scaled: (V, F) or None

    Note:
        - Each feature gets separate StandardScaler (per-column normalization)
        - Fit ONLY on training data to prevent leakage
        - Val and test use same scalers fitted on train
        - Scalers stored in dict: {feature_name: StandardScaler}
    """
    scalers = {}
    train_scaled = np.zeros((len(train_df), len(feature_cols)))
    test_scaled = np.zeros((len(test_df), len(feature_cols)))
    val_scaled = np.zeros((len(val_df), len(feature_cols))) if val_df is not None else None

    for i, col in enumerate(feature_cols):
        scaler = StandardScaler()
        train_scaled[:, i] = np.asarray(scaler.fit_transform(train_df[[col]].values)).reshape(-1)
        test_scaled[:, i] = np.asarray(scaler.transform(test_df[[col]].values)).reshape(-1)
        if val_df is not None and val_scaled is not None:
            val_scaled[:, i] = np.asarray(scaler.transform(val_df[[col]].values)).reshape(-1)
        scalers[col] = scaler

    return train_scaled, test_scaled, val_scaled, scalers


def window(
    features_scaled: np.ndarray,
    target_series: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series prediction.

    Args:
        features_scaled: Scaled feature array (M, F)
        target_series: Target values (M,) - single column
        lookback: Number of historical timesteps (L)
        horizon: Number of future timesteps to predict (K)

    Returns:
        (X, y) where X:(N, L, F) and y:(N, K)

    Shape:
        features_scaled: (M, F)
        target_series: (M,)
        X: (N, L, F) - N sliding windows
        y: (N, K) - Future targets for each window

    Formula:
        N = M - L - K
        For i in range(N):
            X[i] = features[i:i+L, :]        # Past L steps, all F features
            y[i] = target[i+L:i+L+K]         # Future K steps, single target

    Critical:
        - If horizon=1 (single-step), output is (N, 1) NOT (N,)
        - Target must be 1D array (single target column)
    """
    features_scaled = np.asarray(features_scaled, dtype=float)
    target_series = np.asarray(target_series, dtype=float)

    M, F = features_scaled.shape

    if len(target_series.shape) > 1:
        if target_series.shape[1] == 1:
            target_series = target_series.ravel()
        else:
            raise ValueError(f"target_series must be 1D or (N,1), got {target_series.shape}")

    if len(target_series) != M:
        raise ValueError(
            f"Feature and target length mismatch: {M} vs {len(target_series)}"
        )

    # Calculate number of samples
    N = M - lookback - horizon

    if N <= 0:
        raise ValueError(
            f"Not enough data: M={M}, lookback={lookback}, horizon={horizon} → N={N}"
        )

    # Preallocate arrays
    X = np.zeros((N, lookback, F), dtype=np.float32)
    y = np.zeros((N, horizon), dtype=np.float32)

    # Create windows
    for i in range(N):
        X[i] = features_scaled[i : i + lookback, :]
        y[i] = target_series[i + lookback : i + lookback + horizon]

    # Validate shapes
    assert X.shape == (N, lookback, F), f"X shape mismatch: {X.shape}"
    assert y.shape == (N, horizon), f"y shape mismatch: {y.shape}"

    return X, y


# Example usage
if __name__ == "__main__":
    # Demo pipeline
    print("Demo: Data Converters Pipeline")

    # 1. Create dummy data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "open": np.random.randn(100).cumsum() + 100,
            "high": np.random.randn(100).cumsum() + 105,
            "low": np.random.randn(100).cumsum() + 95,
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    print(f"1. Raw data: {df.shape}")

    # 2. Clean
    df = clean_data(df)
    print(f"2. After clean: {df.shape}")

    # 3. Build target (returns Series, align back into df)
    tgt_series, target_col = build_target(df, "close", mode="return", use_log=False)
    df = df.loc[tgt_series.index].copy()
    df[target_col] = tgt_series.values
    print(f"3. After target: {df.shape}, target={target_col}")

    # 4. Split (example usage block at bottom only)
    train_df, test_df, _ = time_split(df, test_size=0.2)
    print(f"4. After split: train={train_df.shape}, test={test_df.shape}")

    # 5. Scale
    feature_cols = ["open", "high", "low", "close", "volume"]
    train_scaled, test_scaled, val_scaled, scalers = scale_features(
        train_df, test_df, feature_cols, val_df=None
    )
    print(f"5. After scale: train={train_scaled.shape}, test={test_scaled.shape}")

    # 6. Window
    target_train = train_df[target_col].to_numpy(dtype=float)
    target_test = test_df[target_col].to_numpy(dtype=float)

    X_train, y_train = window(train_scaled, target_train, lookback=10, horizon=1)
    X_test, y_test = window(test_scaled, target_test, lookback=10, horizon=1)

    print(f"6. After window:")
    print(f"   Train: X{X_train.shape} → y{y_train.shape}")
    print(f"   Test:  X{X_test.shape} → y{y_test.shape}")
    print(f"   Features: {feature_cols}")
    print("✓ Pipeline complete")
