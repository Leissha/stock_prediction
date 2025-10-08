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
from typing import Optional, Tuple, Any
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import warnings
warnings.filterwarnings('ignore')


class SARIMAXModel:
    """
    Model: ARIMA(p, d, q) + seasonal (P, D, Q, m) + optional exogenous regressors (exog).
    How it works? 
        Fit a state-space model and estimate parameters by maximizing likelihood; inference uses the Kalman filter/smoother.
    Endog: the target series (e.g: Close).
    Exog: 2D array of multivariate features aligned in time (flattened lag_days × features window).
    Orders:
        order=(p,d,q): AR lags, differences, MA lags (non-seasonal).
        seasonal_order=(P,D,Q,m): seasonal AR lags, seasonal differences, seasonal MA lags, period m.
    Multistep prediction:
        At each index, forecast k steps using future exogenous rows.
        Then append one observed y to update the state before moving forward.
    """
    def __init__(self, seasonal: bool = False, m: int = 5):
        self.seasonal = bool(seasonal)
        self.m = int(m)
        self.order: Optional[Tuple[int, int, int]] = None # p, d, q
        self.seasonal_order: Optional[Tuple[int, int, int, int]] = None # P, D, Q, m
        self._results = None
        self._n_train: int = 0
        # Always use SARIMA (no SARIMAX exogenous regressors) to avoid leakage of multivariate features

    @staticmethod
    def _flatten_sequences(x):
        # x: (n_samples, lag, n_features) -> (n_samples, lag*n_features)
        if x.ndim != 3:
            raise ValueError("Expected x with 3 dimensions: (samples, lag, features)")
        n, lag, n_feat = x.shape
        return x.reshape(n, lag * n_feat)

    @staticmethod
    def _select_target_series(y):
        # y can be (n,), (n,1), or (n,k) for multistep. Use first horizon for fitting
        if y.ndim == 1:
            return y
        if y.ndim == 2:
            return y[:, 0]
        raise ValueError("Unsupported y dimensions for SARIMAX fitting")

    def fit(self, data=None):
        # Build univariate target series (scaled if scaler is available). No exog to avoid leakage
        if data is None:
            raise ValueError("fit requires 'data' dict with train_df and target_feature to build SARIMA series")
        target_feature = data.get('target_feature')
        train_df = data.get('train_df')
        scalers = data.get('scalers', {})
        if train_df is None or target_feature is None:
            raise ValueError("data must include 'train_df' and 'target_feature'")
        target_col = target_feature
        scaler = None
        for k, v in scalers.items():
            if k.endswith(f"_{target_col}"):
                scaler = v
                break
        series = train_df[target_col].astype(float).values.reshape(-1, 1)
        if scaler is not None:
            series_scaled = scaler.transform(series).reshape(-1)
        else:
            series_scaled = series.reshape(-1)
        target_series = series_scaled

        # Auto-order selection
        auto = pm.auto_arima(
            y=target_series,
            seasonal=self.seasonal,
            m=self.m if self.seasonal else 1,
            start_p=0,
            start_q=0,
            max_p=3,
            max_q=3,
            max_d=2,
            start_P=0,
            start_Q=0,
            max_P=2,
            max_Q=2,
            max_D=1,
            stepwise=True,
            error_action='ignore',
            suppress_warnings=True,
            trace=False,
        )

        self.order = tuple(auto.order)  # type: ignore
        self.seasonal_order = tuple(auto.seasonal_order) if self.seasonal else (0, 0, 0, 0)  # type: ignore

        model = SARIMAX(target_series, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        self._results = model.fit(disp=False)
        self._n_train = len(target_series)
        return self

    def predict_and_evaluate(self, x_test, y_test):
        if self._results is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # No exog in SARIMA
        exog_regressors_test = None

        # Determine multistep horizon from y_test
        # Normalise y_test shape from various sequence creators to a consistent form
        # Accepted inputs:
        #   (N,)           -> single-step
        #   (N, 1)         -> single-step
        #   (N, K)         -> K-step horizons
        #   (N, 1, 1)      -> squeeze to (N,)
        #   (N, 1, K)      -> squeeze to (N, K)
        y_test = np.asarray(y_test)
        # Support shapes: (N,), (N,1), (N,K) and also (N,1,1)
        if y_test.ndim == 3 and y_test.shape[1] == 1 and y_test.shape[2] == 1:
            y_test = y_test.reshape(y_test.shape[0])  # -> (N,)
        elif y_test.ndim == 3 and y_test.shape[1] == 1:
            y_test = y_test[:, 0, :]  # -> (N, K)

        if y_test.ndim == 1:
            k = 1
        elif y_test.ndim == 2:
            k = y_test.shape[1]
        else:
            raise ValueError("Unsupported y_test dimensions for prediction")

        n_sequences = x_test.shape[0]
        predictions = np.zeros((n_sequences, k), dtype=float)

        # Rolling forecasts: at each index, forecast k steps using future exogenous rows
        # Then append one observed y to update the state before moving forward
        res: Any = self._results
        idx = 0
        while idx < n_sequences:
            # Multi-step forecast (SARIMA: exog=None). If exog were used and known, pass next k rows.
            k_slice = None
            fc = res.get_forecast(steps=k, exog=k_slice)
            yhat = fc.predicted_mean
            predictions[idx, :] = yhat

            # Append one actual observation to update state for next roll (use first horizon)
            if y_test.ndim == 2:
                y_obs = y_test[idx, 0]
            else:
                y_obs = y_test[idx]
            # Append expects arrays, not scalars
            res = res.append(endog=np.array([y_obs]), exog=None)
            idx += 1

        # Truncate to the number of valid sequences (test harness handles remainder)
        valid_n = idx
        predictions = predictions[:valid_n, :]

        # Dummy metrics for interface compatibility (loss not applicable here)
        metrics = {
            'loss': 0.0,
            'mae': float('nan'),
            'rmse': float('nan'),
            'loss_name': 'na',
            'mae_name': 'mean_absolute_error',
            'rmse_name': 'root_mean_squared_error'
        }

        return predictions, metrics