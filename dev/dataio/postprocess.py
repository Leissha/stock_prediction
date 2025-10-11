"""
Post-prediction processing: align, descale, convert to prices.

All functions maintain (N, K) shape contracts.
"""

from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler


def descale(arr: np.ndarray, scaler: Optional[StandardScaler]) -> np.ndarray:
    """
    Inverse StandardScaler transformation.

    Args:
        arr: Scaled array (N, K)
        scaler: Fitted StandardScaler (or None for no-op)

    Returns:
        Unscaled array (N, K)

    Note:
        StandardScaler expects 2D input with features as columns.
        For (N, K) target arrays, we treat each K as a separate feature.
    """
    if scaler is None:
        return np.asarray(arr, dtype=float)

    if not hasattr(scaler, "inverse_transform"):
        print("WARNING: Scaler has no inverse_transform method")
        return np.asarray(arr, dtype=float)

    arr = np.asarray(arr)
    original_shape = arr.shape

    # Ensure 2D
    if arr.ndim == 1:
        arr = np.reshape(arr, (-1, 1))

    # Inverse transform
    try:
        result = scaler.inverse_transform(arr)
        result = np.asarray(result)
    except Exception as e:
        print(f"WARNING: Scaler inverse_transform failed: {e}")
        result = np.asarray(arr, dtype=float)

    # Restore original shape if needed
    if result.shape != original_shape:
        try:
            result = np.reshape(result, original_shape)
        except Exception:
            pass

    return np.asarray(result)


def returns_to_prices(
    returns: np.ndarray,
    base_prices: np.ndarray,
    mode: str,
    use_log: bool = False,
    validate: bool = True,
) -> np.ndarray:
    """
    Convert returns to prices using per-row base prices.

    Args:
        returns: Return predictions (N, K) or prices if mode="price"
        base_prices: Base price for each sample (N,) - price at time t
        mode: "price" | "return" | "log_return"
        use_log: If True, returns are log returns
        validate: If True, check for extreme values

    Returns:
        Prices (N, K)

    Formulas:
        - price: return input unchanged
        - return: price[i,k] = base[i] * prod(1 + r[i,0:k+1])
        - log_return: price[i,k] = base[i] * prod(exp(r[i,0:k+1]))

    Example:
        base_prices = [100, 101, 102]  # Price at time t for each sample
        returns = [[0.02, 0.01], [0.01, -0.01], [-0.01, 0.02]]  # (3, 2)

        For sample 0:
            price[0,0] = 100 * (1 + 0.02) = 102.0
            price[0,1] = 102 * (1 + 0.01) = 103.02

        For sample 1:
            price[1,0] = 101 * (1 + 0.01) = 102.01
            price[1,1] = 102.01 * (1 - 0.01) = 100.99
    """
    # Ensure 2D
    if returns.ndim == 1:
        returns = returns[:, None]

    N, K = returns.shape

    # Ensure base_prices is 1D with length N
    base_prices = np.asarray(base_prices).ravel()
    if len(base_prices) != N:
        raise ValueError(f"base_prices length {len(base_prices)} != N {N}")

    # Price mode: no conversion needed
    if mode == "price":
        return returns.astype(float)

    # Validate returns if requested
    if validate:
        if use_log or mode == "log_return":
            # Log returns typically in range [-0.5, 0.5] for daily data
            if np.any(np.abs(returns) > 1.0):
                max_val = np.max(np.abs(returns))
                print(f"WARNING: Large log returns detected: max={max_val:.4f}")
        else:
            # Simple returns: -1 means price → 0, which is extreme
            if np.any(returns < -0.5):
                min_val = np.min(returns)
                print(f"WARNING: Large negative returns detected: min={min_val:.4f}")

    # Compound returns to prices
    prices = np.zeros((N, K), dtype=float)

    for sample_idx in range(N):
        current_price = float(base_prices[sample_idx])

        for step in range(K):
            if use_log or mode == "log_return":
                # Log return: multiply by exp(r)
                current_price *= np.exp(float(returns[sample_idx, step]))
            else:
                # Simple return: multiply by (1 + r)
                current_price *= 1.0 + float(returns[sample_idx, step])

            prices[sample_idx, step] = current_price

    # Validate output
    if validate:
        if np.any(prices <= 0):
            print("WARNING: Negative or zero prices detected after conversion")
            print(f"  Min price: {np.min(prices):.4f}")
            print(f"  Zero count: {np.sum(prices == 0)}")

    return prices


def prices_to_returns(
    prices: np.ndarray, base_prices: np.ndarray, mode: str = "return", use_log: bool = False
) -> np.ndarray:
    """
    Convert prices to returns (inverse of convert_returns_to_prices).

    Args:
        prices: Price predictions (N, K)
        base_prices: Base price for each sample (N,)
        mode: "return" | "log_return"
        use_log: If True, compute log returns

    Returns:
        Returns (N, K)

    Note:
        For multi-step (K>1), returns are NOT independent - they compound.
        This function computes sequential returns: r[k] = (price[k] - price[k-1]) / price[k-1]
    """
    if prices.ndim == 1:
        prices = prices[:, None]

    N, K = prices.shape
    base_prices = np.asarray(base_prices).ravel()

    if len(base_prices) != N:
        raise ValueError(f"base_prices length {len(base_prices)} != N {N}")

    returns = np.zeros((N, K), dtype=float)

    for sample_idx in range(N):
        prev_price = float(base_prices[sample_idx])

        for step in range(K):
            curr_price = float(prices[sample_idx, step])

            if use_log or mode == "log_return":
                returns[sample_idx, step] = np.log(curr_price / prev_price)
            else:
                returns[sample_idx, step] = (curr_price - prev_price) / prev_price

            prev_price = curr_price

    return returns


def validate_price_conversion(
    returns: np.ndarray, base_prices: np.ndarray, mode: str, use_log: bool
) -> None:
    """
    Test return→price→return roundtrip.

    Args:
        returns: Original returns (N, K)
        base_prices: Base prices (N,)
        mode: "return" | "log_return"
        use_log: Whether log returns

    Raises:
        AssertionError: If roundtrip doesn't match
    """
    # Forward: returns → prices
    prices = returns_to_prices(returns, base_prices, mode, use_log, validate=False)

    # Backward: prices → returns
    returns_back = prices_to_returns(prices, base_prices, mode, use_log)

    # Check match
    max_diff = np.max(np.abs(returns - returns_back))
    if max_diff > 1e-6:
        raise AssertionError(f"Roundtrip error: max diff = {max_diff:.2e}")

    print(f"✓ Roundtrip validation passed (max diff = {max_diff:.2e})")


# Example usage
if __name__ == "__main__":
    print("Smoke test: Postprocessing Pipeline")

    # 1. Create dummy predictions and ground truth
    N, K = 10, 3
    np.random.seed(42)

    # Simulate scaled predictions
    y_pred_scaled = np.random.randn(N, K) * 0.5  # Scaled returns
    y_true_scaled = np.random.randn(N, K) * 0.5

    print(f"1. Raw predictions: y_pred{y_pred_scaled.shape}, y_true{y_true_scaled.shape}")

    # 2. Descale (simulate StandardScaler inverse)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    # Fit on some dummy data
    dummy = np.random.randn(100, K) * 0.02 + 0.01  # Returns centered at 1%
    scaler.fit(dummy)

    y_pred_descaled = descale(y_pred_scaled, scaler)
    y_true_descaled = descale(y_true_scaled, scaler)

    print(f"2. After descale: ranges pred=[{y_pred_descaled.min():.4f}, {y_pred_descaled.max():.4f}]")

    # 3. Convert to prices
    base_prices = np.linspace(100, 110, N)  # Per-row base prices

    y_pred_prices = returns_to_prices(
        y_pred_descaled, base_prices, mode="return", use_log=False
    )
    y_true_prices = returns_to_prices(
        y_true_descaled, base_prices, mode="return", use_log=False
    )

    print(f"3. After returns_to_prices:")
    print(f"   Pred: {y_pred_prices[0]}")
    print(f"   True: {y_true_prices[0]}")
    print(f"   Base: {base_prices[0]}")

    # 4. Validate roundtrip
    test_returns = np.array([[0.02, 0.01, -0.01], [0.01, 0.015, 0.02]])
    test_base = np.array([100.0, 105.0])

    print("\n4. Roundtrip validation:")
    validate_price_conversion(test_returns, test_base, mode="return", use_log=False)
    validate_price_conversion(
        np.log(1 + test_returns), test_base, mode="log_return", use_log=True
    )

    # 5. Compare simple vs log returns
    print("\n5. Simple vs Log Returns (same base=100, r=0.05):")
    test_return_simple = np.array([[0.05]])
    test_return_log = np.array([[np.log(1.05)]])
    test_base_single = np.array([100.0])

    price_simple = returns_to_prices(test_return_simple, test_base_single, "return", False)
    price_log = returns_to_prices(test_return_log, test_base_single, "log_return", True)

    print(f"   Simple return 0.05 → price {price_simple[0,0]:.2f}")
    print(f"   Log return {np.log(1.05):.4f} → price {price_log[0,0]:.2f}")
    print(f"   Both should be 105.0")
