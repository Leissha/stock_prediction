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
    
    def __post_init__(self):
        # Automatically detect multistep mode based on lookup_steps
        self.multistep_mode = self.lookup_steps > 1


