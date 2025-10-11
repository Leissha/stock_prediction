import numpy as np
from typing import Dict, Optional


def compute_metrics(y_true_px: np.ndarray, y_hat_px: np.ndarray, epochs_trained: Optional[int] = None) -> Dict[str, float]:
    """
    Compute MAE, RMSE, DA@1 in price space.

    Expected inputs: (N, K) each.

    Args:
        y_true_px: Ground truth prices (N, K)
        y_hat_px: Predicted prices (N, K)
        epochs_trained: Optional number of epochs actually trained (for early stopping info)

    Returns:
        Dictionary with metrics (mae, rmse, directional_accuracy, and optionally epochs_trained)
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

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'directional_accuracy': da,
    }

    # Add epochs_trained if provided
    if epochs_trained is not None:
        metrics['epochs_trained'] = epochs_trained

    return metrics


