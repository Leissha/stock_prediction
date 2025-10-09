import numpy as np
from typing import Dict


def compute_metrics(y_true_px: np.ndarray, y_hat_px: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, DA@1 in price space.

    Expected inputs: (N, K) each.
    """
    # Flatten for metrics (take first step if K>1)
    y_true_flat = y_true_px[:, 0]
    y_hat_flat = y_hat_px[:, 0]

    # MAE, RMSE
    mae = float(np.mean(np.abs(y_true_flat - y_hat_flat)))
    rmse = float(np.sqrt(np.mean((y_true_flat - y_hat_flat) ** 2)))

    # DA@1: directional accuracy for first step
    if y_true_px.shape[0] > 1:
        true_dir = np.sign(y_true_flat[1:] - y_true_flat[:-1])
        pred_dir = np.sign(y_hat_flat[1:] - y_true_flat[:-1])
        da = float(np.mean(true_dir == pred_dir))
    else:
        da = 0.5

    return {
        'mae': mae,
        'rmse': rmse,
        'directional_accuracy': da,
    }


