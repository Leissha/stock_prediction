from dataclasses import dataclass
import os


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model (weighted average only)."""
    sarima_weight: float = 0.5
    lstm_weight: float = 0.5
    normalize_weights: bool = True


class EnsembleModel:
    """
    Ensemble that combines SARIMA and LSTM predictions via weighted average.
    Models must be pre-fitted and expose predict() that returns (N, K).
    """

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.sarima_model = None
        self.lstm_model = None
        self.is_fitted = False

        # Validate and normalize weights
        if self.config.normalize_weights:
            total_weight = self.config.sarima_weight + self.config.lstm_weight
            if total_weight > 0:
                self.config.sarima_weight /= total_weight
                self.config.lstm_weight /= total_weight
            else:
                raise ValueError("At least one weight must be positive")
    
    def fit(self, sarima_model, lstm_model):
        """
        Register pre-trained models.

        Args:
            sarima_model: Pre-trained SARIMA model with predict(X_test) -> (N,K)
            lstm_model: Pre-trained LSTM model with predict(X_test) -> (N,K)
        """
        self.sarima_model = sarima_model
        self.lstm_model = lstm_model
        self.is_fitted = True
    
    def predict(self, x_test):
        """Return weighted-average predictions with shape (N, K)."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble model not fitted. Call fit() first.")

        if self.sarima_model is None or self.lstm_model is None:
            raise ValueError("Both SARIMA and LSTM models must be set")

        sarima_preds = self.sarima_model.predict(x_test)
        lstm_preds = self.lstm_model.predict(x_test)

        # Strict shape validation (no silent fixing)
        if isinstance(sarima_preds, tuple) or isinstance(lstm_preds, tuple):
            raise TypeError("Submodels must return numpy arrays, not tuples")
        if sarima_preds.ndim != 2 or lstm_preds.ndim != 2:
            raise ValueError(
                f"Predictions must be 2D (N,K). Got SARIMA {sarima_preds.shape}, LSTM {lstm_preds.shape}"
            )
        if sarima_preds.shape != lstm_preds.shape:
            raise ValueError(
                f"Prediction shape mismatch: SARIMA {sarima_preds.shape} vs LSTM {lstm_preds.shape}"
            )

        # Weighted average
        ensemble_preds = (
            self.config.sarima_weight * sarima_preds +
            self.config.lstm_weight * lstm_preds
        )
        return ensemble_preds

    def get_model_info(self):
        """Get information about the ensemble model."""
        return {
            'sarima_weight': self.config.sarima_weight,
            'lstm_weight': self.config.lstm_weight,
            'is_fitted': self.is_fitted,
        }

    def fit_from_bundle(self, bundle, sarima_args: dict, lstm_args: dict, lstm_model_path: str):
        """
        Assemble and prepare submodels from the DataBundle.

        Args:
            bundle: DataBundle with train/val splits and metadata
            sarima_args: { 'seasonal': bool, 'm': int }
            lstm_args: { 'layers': List[int], 'dropout_rate': float, 'epochs': int, 'batch_size': int, 'model_name': str='lstm' }
            lstm_model_path: filesystem path for saving/loading the LSTM model
        """
        # Import locally to avoid circular imports at module load time
        from model.sarimax import SARIMAXModel
        from model.tf_models import TFModel
        from tensorflow import keras as tf_keras  # type: ignore

        # 1) SARIMA from bundle (unscaled target)
        sarima = SARIMAXModel(
            seasonal=bool(sarima_args.get('seasonal', False)),
            m=int(sarima_args.get('m', 5)),
        )
        data_dict = {
            'train_df': bundle.train_df,
            'target_feature': bundle.target_feature,
            'scalers': bundle.scalers,
            'test_df': bundle.test_df,
        }
        sarima.fit(data=data_dict)

        # 2) LSTM from bundle (load or train)
        input_size = bundle.X_train.shape[2]
        lookup_steps = bundle.horizon
        lstm = TFModel(
            input_size=input_size,
            model_name=str(lstm_args.get('model_name', 'lstm')),
            layers=lstm_args.get('layers'),
            dropout_rate=float(lstm_args.get('dropout_rate', 0.2)),
            output_steps=lookup_steps,
        )
        if not os.path.exists(lstm_model_path):
            val_tuple = None
            if bundle.X_val is not None and bundle.y_val is not None:
                val_tuple = (bundle.X_val, bundle.y_val)
            lstm.fit(
                bundle.X_train,
                bundle.y_train,
                epochs=int(lstm_args.get('epochs', 25)),
                batch_size=int(lstm_args.get('batch_size', 32)),
                validation_data=val_tuple,
            )
            lstm.save_model(lstm_model_path)
        else:
            keras_model = tf_keras.models.load_model(lstm_model_path)
            lstm.model = keras_model

        # 3) Register
        self.fit(sarima, lstm)
        return self
