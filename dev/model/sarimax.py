"""
ARIMA/SARIMA Model Implementation for Time Series Forecasting
References:
- how it works: https://youtu.be/drlt0pNEUH4?si=rsm6GqGsxmFVhXE5 
- code guide: https://medium.com/@injure21/arima-for-anomaly-detection-85bfdef5d585
- log, pacf plots explanation: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
- code guide: https://github.com/liannewriting/YouTube-videos-public/blob/main/arima-model-time-series-prediction-python/time-series-arima.ipynb 
- https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
- https://blog.quantinsti.com/forecasting-stock-returns-using-arima-model/
https://www.kaggle.com/code/nageshsingh/stock-market-forecasting-arima 
https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/ 

ARIMAX/SARIMAX implementation compatible with our training pipeline.

Notes:
- Input sequences come as 3D tensors (n_samples, lag_days, n_features) already scaled.
- We flatten each sequence window into exogenous regressors: shape (n_samples, lag_days * n_features).
- The target is a single output variable (e.g., Close). For multistep training targets shaped (N, lookup_steps), we use the first horizon to fit the time-series model and generate k-step (lookup_steps) forecasts at predict time using exogenous inputs.
"""

import numpy as np
from typing import Optional, Tuple, cast
import warnings
warnings.filterwarnings('ignore')


class SARIMAXModel:
    """
    Model: ARIMA(p, d, q) + seasonal (P, D, Q, m) with target feature as endog (currently no OHLCV exog to avoid leakage)
    How it works? 
        Fit a state-space model and estimate parameters by maximizing likelihood; inference uses the Kalman filter/smoother.
    Endog: the target series (e.g: Close).
    Exog: 2D array of multivariate features aligned in time (flattened lag_days × features window).
    Orders:
        order=(p,d,q): AR lags, differences, MA lags (non-seasonal).
        seasonal_order=(P,D,Q,m): seasonal AR lags, seasonal differences, seasonal MA lags, period m.
    Multistep prediction:
        At each index, forecast k steps using future rows.
        Then append one observed y to update the state before moving forward.
    """
    def __init__(self, seasonal: bool = False, m: int = 5, forecast_horizon: int = 1):
        self.seasonal = bool(seasonal)
        self.m = int(m)
        self.forecast_horizon = int(forecast_horizon)
        self.order: Optional[Tuple[int, int, int]] = None # p, d, q
        self.seasonal_order: Optional[Tuple[int, int, int, int]] = None # P, D, Q, m
        self._results = None
        self._n_train: int = 0
        self._target_scaler = None
        # Note: Uses ARIMA/SARIMA without exogenous regressors to avoid leakage

    def fit(self, data=None):
        """Fit SARIMAX using the unscaled target series from the bundle."""
        if data is None:
            raise ValueError("fit requires 'data' with 'train_df' and 'target_feature'")
        target_feature = data.get('target_feature')
        train_df = data.get('train_df')
        if train_df is None or target_feature is None:
            raise ValueError("data must include 'train_df' and 'target_feature'")

        # Always use unscaled target provided by the pipeline/bundle
        target_series = train_df[target_feature].astype(float).values
        # Use AutoARIMA for parameter selection, then fit with statsmodels
        from statsforecast.models import AutoARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
        import pandas as pd
        
        # Use statsforecast.AutoARIMA for fast parameter selection
        y_series = pd.Series(target_series, name='y')
        
        # Auto-select order/seasonal_order via statsforecast.AutoARIMA
        auto_arima = AutoARIMA(
            season_length=self.m if self.seasonal else 1,
            max_p=3, max_q=3,
            max_P=2, max_Q=2,
            max_d=2,
            max_D=1 if self.seasonal else 0,
            start_p=1, start_q=1,
            stepwise=True,
            approximation=False,
            seasonal=self.seasonal,
            trace=False
        )
        auto_arima.fit(y_series.to_numpy(dtype=float))
        arma = auto_arima.model_['arma']
        self.order = (arma[0], arma[1], arma[2])
        seasonal_p, seasonal_d, seasonal_q, seasonal_m = arma[3], arma[4], arma[5], arma[6]
        if seasonal_m == 0 and (seasonal_p > 0 or seasonal_d > 0 or seasonal_q > 0):
            seasonal_m = self.m if self.seasonal else 5
        elif seasonal_m == 1:
            seasonal_m = 5
        self.seasonal_order = (seasonal_p, seasonal_d, seasonal_q, seasonal_m)
        
        
        # Fit statsmodels.SARIMAX with selected parameters
        model = SARIMAX(
            target_series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self._results = cast(SARIMAXResults, model.fit(disp=False))
        self._n_train = len(target_series)
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Generate rolling forecasts.

        Args:
            x_test: shape (N, L, F) - only N is used, L and F ignored

        Returns:
            predictions: shape (N, K) where K = forecast_horizon
        """
        if self._results is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        N = x_test.shape[0]
        K = self.forecast_horizon

        from statsmodels.tsa.statespace.sarimax import SARIMAXResults
        res = cast(SARIMAXResults, self._results)
        preds = np.zeros((N, K))
        for i in range(N):
            fc = res.get_forecast(steps=K)
            m = np.asarray(fc.predicted_mean)
            preds[i, :] = m[:K]
            # walk-forward update with own 1-step forecast as proxy
            res = cast(SARIMAXResults, res.append(endog=np.array([m[0]])))
        return preds