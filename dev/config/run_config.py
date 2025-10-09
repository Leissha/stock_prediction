from dataclasses import dataclass
from typing import List


@dataclass
class RunConfig:
    ticker: str
    start_date: str
    end_date: str
    target_feature: str
    lag_days: int
    test_size: float
    val_size: float
    split_method: str
    shuffle: bool
    scale: bool
    model_name: str
    layers: List[int]
    dropout_rate: float
    epochs: int
    batch_size: int
    target_as_return: bool
    use_log_returns: bool
    base_path: str
    data_path: str
    model_path: str
    plot_path: str
    report_path: str
    # Multistep prediction parameters
    lookup_steps: int = 1
    # SARIMAX params
    sarimax_seasonal: bool = False
    sarimax_m: int = 5
    # Ensemble params
    ensemble_method: str = 'weighted_average'  # 'weighted_average', 'stacking', 'voting'
    sarima_weight: float = 0.2
    lstm_weight: float = 0.8


