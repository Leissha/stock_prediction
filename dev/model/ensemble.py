from dataclasses import dataclass
import os


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model (weighted average only)."""
    sarima_weight: float = 0.5
    model_2_weight: float = 0.5
    normalize_weights: bool = True


class EnsembleModel:
    """
    Ensemble that combines SARIMA and LSTM predictions via weighted average.
    Models must be pre-fitted and expose predict() that returns (N, K).
    """

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.sarima_model = None
        self.model_2 = None
        self.is_fitted = False

        # Validate and normalize weights
        if self.config.normalize_weights:
            total_weight = self.config.sarima_weight + self.config.model_2_weight
            if total_weight > 0:
                self.config.sarima_weight /= total_weight
                self.config.model_2_weight /= total_weight
            else:
                raise ValueError("At least one weight must be positive")
    
    def fit(self, sarima_model, model_2):
        """
        Register pre-trained models.

        Args:
            sarima_model: Pre-trained SARIMA model with predict(X_test) -> (N,K)
            model_2: Pre-trained TF model with predict(X_test) -> (N,K)
        """
        self.sarima_model = sarima_model
        self.model_2 = model_2
        self.is_fitted = True
    
    def predict(self, x_test):
        """Return weighted-average predictions with shape (N, K)."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble model not fitted. Call fit() first.")

        if self.sarima_model is None or self.model_2 is None:
            raise ValueError("Both SARIMA and Model 2 models must be set")

        sarima_preds = self.sarima_model.predict(x_test)
        model_2_preds = self.model_2.predict(x_test)

        # Shape validation
        if isinstance(sarima_preds, tuple) or isinstance(model_2_preds, tuple):
            raise TypeError("Submodels must return numpy arrays, not tuples")
        if sarima_preds.ndim != 2 or model_2_preds.ndim != 2:
            raise ValueError(
                f"Predictions must be 2D (N,K). Got SARIMA {sarima_preds.shape}, Model 2 {model_2_preds.shape}"
            )
        if sarima_preds.shape != model_2_preds.shape:
            raise ValueError(
                f"Prediction shape mismatch: SARIMA {sarima_preds.shape} vs Model 2 {model_2_preds.shape}"
            )

        # Weighted average
        ensemble_preds = (
            self.config.sarima_weight * sarima_preds +
            self.config.model_2_weight * model_2_preds
        )
        return ensemble_preds

    def fit_from_bundle(self, bundle, sarima_args: dict, model_2_args: dict, model_2_meta_path: str):
        """
        Assemble and prepare submodels from the DataBundle.

        Args:
            bundle: DataBundle with train/val splits and metadata
            sarima_args: { 'seasonal': bool, 'm': int }
            model_2_args: { 'layers': List[int], 'dropout_rate': float, 'epochs': int, 'batch_size': int, 'model_name': str='lstm' }
            model_2_meta_path: filesystem path for saving/loading the TF model
        """
        # Import locally to avoid circular imports at module load time
        from model.sarimax import SARIMAXModel
        from model.tf_models import TFModel
        from tensorflow import keras as tf_keras  # type: ignore

        # 1) SARIMA from bundle (unscaled target)
        sarima = SARIMAXModel(
            seasonal=bool(sarima_args.get('seasonal', False)),
            m=int(sarima_args.get('m', 5)),
            forecast_horizon=bundle.horizon,
        )
        data_dict = {
            'train_df': bundle.train_df,
            'target_feature': bundle.target_feature,
            'scalers': bundle.scalers,
            'test_df': bundle.test_df,
        }
        sarima.fit(data=data_dict)

        # 2) Model 2 from bundle (load or train)
        model_2 = TFModel(
            input_size=bundle.X_train.shape[2],
            model_name=str(model_2_args.get('model_name', 'lstm')),
            layers=model_2_args.get('layers'),
            dropout_rate=float(model_2_args.get('dropout_rate', 0.2)),
            output_steps=bundle.horizon,
        )
        model_2_model_path = f"cache/trained_models/{model_2_meta_path}.keras"
        if not os.path.exists(model_2_model_path):
            val_tuple = None
            if bundle.X_val is not None and bundle.y_val is not None:
                val_tuple = (bundle.X_val, bundle.y_val)
                
            fit_kwargs = {
                'epochs': int(model_2_args.get('epochs', 25)),
                'batch_size': int(model_2_args.get('batch_size', 32)),
                'validation_data': val_tuple,
                'optimizer': str(model_2_args.get('optimizer', 'adam')),
                'learning_rate': float(model_2_args.get('learning_rate', 1e-3)),
                'meta_path': model_2_meta_path,
            }

            model_2.fit(
                bundle.X_train,
                bundle.y_train,
                **fit_kwargs,
            )
            model_2.save_model(model_2_model_path)
        else:
            keras_model = tf_keras.models.load_model(model_2_model_path)
            model_2.model = keras_model

        # 3) Register
        self.fit(sarima, model_2)
        return self
