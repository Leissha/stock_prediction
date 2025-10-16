from typing import List, Optional, Dict, Any
import pandas as pd
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, model_validator

NDArray = np.ndarray

class TargetMode(str, Enum):
    PRICE = "price"
    RETURN = "return"
    LOG_RETURN = "log_return"

class DataBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required train/test data
    X_train: NDArray  # Shape: (N, L, F)
    y_train: NDArray  # Shape: (N, K)
    X_test: NDArray   # Shape: (N, L, F)
    y_test: NDArray   # Shape: (N, K)

    # Optional validation data
    X_val: Optional[NDArray] = None
    y_val: Optional[NDArray] = None
    base_prices_val: Optional[NDArray] = None

    # Metadata
    symbol: str
    features: List[str]
    lookback: int = Field(gt=0)
    horizon: int = Field(gt=0)
    target_mode: TargetMode
    use_log_returns: bool = False
    scalers: Dict[str, Any] = Field(default_factory=dict)
    target_scaler: Optional[Any] = None
    base_prices_test: Optional[NDArray] = None

    # Sentiment integration (optional)
    use_sentiment: bool = False

    # Optional fields for SARIMAX (original DataFrames before sequencing)
    train_df: Optional[Any] = None  # pd.DataFrame
    test_df: Optional[Any] = None   # pd.DataFrame
    val_df: Optional[Any] = None    # pd.DataFrame
    target_feature: Optional[str] = None

    @model_validator(mode='after')
    def validate_all(self):
        # ---- Ranks ----
        if self.X_train.ndim != 3:
            raise ValueError(f"X_train must be 3D (N,L,F), got shape {self.X_train.shape}")
        if self.X_test.ndim != 3:
            raise ValueError(f"X_test must be 3D (N,L,F), got shape {self.X_test.shape}")
        if self.y_train.ndim != 2:
            raise ValueError(f"y_train must be 2D (N,K), got shape {self.y_train.shape}")
        if self.y_test.ndim != 2:
            raise ValueError(f"y_test must be 2D (N,K), got shape {self.y_test.shape}")

        # ---- Sample counts ----
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError(f"N mismatch in train: X_train {self.X_train.shape[0]} vs y_train {self.y_train.shape[0]}")
        if self.X_test.shape[0] != self.y_test.shape[0]:
            raise ValueError(f"N mismatch in test: X_test {self.X_test.shape[0]} vs y_test {self.y_test.shape[0]}")

        # Validation data (optional) - only check if provided
        if self.X_val is not None and self.y_val is not None:
            if self.X_val.shape[0] != self.y_val.shape[0]:
                raise ValueError(f"N mismatch in val: X_val {self.X_val.shape[0]} vs y_val {self.y_val.shape[0]}")

        # ---- Dimensions: lookback / features / horizon ----
        if self.X_train.shape[1] != self.lookback:
            raise ValueError(f"Lookback mismatch (train): X_train.shape[1]={self.X_train.shape[1]} vs lookback={self.lookback}")
        if self.X_test.shape[1] != self.lookback:
            raise ValueError(f"Lookback mismatch (test): X_test.shape[1]={self.X_test.shape[1]} vs lookback={self.lookback}")

        if self.X_train.shape[2] != len(self.features):
            raise ValueError(f"Feature count mismatch (train): X_train.shape[2]={self.X_train.shape[2]} vs len(features)={len(self.features)}")
        if self.X_test.shape[2] != len(self.features):
            raise ValueError(f"Feature count mismatch (test): X_test.shape[2]={self.X_test.shape[2]} vs len(features)={len(self.features)}")

        if self.y_train.shape[1] != self.horizon:
            raise ValueError(f"Horizon mismatch (train): y_train.shape[1]={self.y_train.shape[1]} vs horizon={self.horizon}")
        if self.y_test.shape[1] != self.horizon:
            raise ValueError(f"Horizon mismatch (test): y_test.shape[1]={self.y_test.shape[1]} vs horizon={self.horizon}")

        # ---- Base prices ----
        if self.base_prices_test is not None:
            if self.base_prices_test.ndim != 1:
                raise ValueError(f"base_prices_test must be 1D, got shape {self.base_prices_test.shape}")
            if len(self.base_prices_test) != self.X_test.shape[0]:
                raise ValueError(
                    f"base_prices_test length {len(self.base_prices_test)} != N_test {self.X_test.shape[0]} "
                    f"(expected slice: df[col].iloc[lookback-1 : lookback-1+N_test])"
                )
        
        # Validation base prices
        if self.base_prices_val is not None:
            if self.base_prices_val.ndim != 1:
                raise ValueError(f"base_prices_val must be 1D, got shape {self.base_prices_val.shape}")
            if self.X_val is not None and len(self.base_prices_val) != self.X_val.shape[0]:
                raise ValueError(
                    f"base_prices_val length {len(self.base_prices_val)} != N_val {self.X_val.shape[0]}"
                )

        # ---- Enum/flag consistency ----
        if self.target_mode == TargetMode.LOG_RETURN and not self.use_log_returns:
            raise ValueError("use_log_returns must be True when target_mode=LOG_RETURN")
        if self.target_mode in (TargetMode.PRICE, TargetMode.RETURN) and self.use_log_returns:
            raise ValueError("use_log_returns must be False when target_mode=PRICE or RETURN")

        # ---- target_scaler interface (if provided) ----
        if self.target_scaler is not None and not hasattr(self.target_scaler, "inverse_transform"):
            raise ValueError("target_scaler must implement inverse_transform(arr)")

        # ---- SARIMAX fields consistency ----
        if self.train_df is not None and self.target_feature is not None:
            if self.target_feature not in self.train_df.columns:
                raise ValueError(
                    f"target_feature '{self.target_feature}' not in train_df.columns: {list(self.train_df.columns)}"
                )
        
        # ---- Scaler consistency validation ----
        if self.scalers:
            # Ensure all feature scalers are present
            missing_scalers = set(self.features) - set(self.scalers.keys())
            if missing_scalers:
                raise ValueError(f"Missing scalers for features: {missing_scalers}")
            
            # Ensure target scaler is consistent with target_mode
            if self.target_scaler is not None and "__target__" not in self.scalers:
                raise ValueError("target_scaler provided but not in scalers dict")
            if "__target__" in self.scalers and self.target_scaler is None:
                raise ValueError("__target__ in scalers but target_scaler is None")
        
        # ---- Data consistency across splits (single source of truth) ----
        # Ensure all splits have same feature count
        if self.X_val is not None:
            if self.X_val.shape[2] != self.X_train.shape[2]:
                raise ValueError(f"Feature count mismatch: X_train {self.X_train.shape[2]} vs X_val {self.X_val.shape[2]}")
            if self.X_val.shape[1] != self.lookback:
                raise ValueError(f"Lookback mismatch (val): X_val.shape[1]={self.X_val.shape[1]} vs lookback={self.lookback}")
            if self.y_val is not None and self.y_val.shape[1] != self.horizon:
                raise ValueError(f"Horizon mismatch (val): y_val.shape[1]={self.y_val.shape[1]} vs horizon={self.horizon}")

        return self

    def summary(self) -> str:
        val_info = f"  Val  : X{self.X_val.shape} → y{self.y_val.shape}\n" if self.X_val is not None and self.y_val is not None else ""
        return (
            f"DataBundle[{self.symbol}]\n"
            f"  Train: X{self.X_train.shape} → y{self.y_train.shape}\n"
            f"{val_info}"
            f"  Test : X{self.X_test.shape} → y{self.y_test.shape}\n"
            f"  Mode : {self.target_mode.value} (log={self.use_log_returns})\n"
            f"  Feats: {len(self.features)} -> {', '.join(self.features)}\n"
        )
