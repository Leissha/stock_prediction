"""
TensorFlow Model for lstm, gru, rnn, bilstm
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD  # type: ignore
from loguru import logger
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional  # type: ignore
from typing import List, Optional
from config.data import MODEL_NAME, LAYERS, DROPOUT

# Layer type mapping
_LAYER = {
    "lstm": LSTM,
    "gru": GRU, 
    "rnn": SimpleRNN,
    "bilstm": LSTM
}

class TFModel:
    def __init__(self, input_size: int, model_name: str = MODEL_NAME, layers: Optional[List[int]] = LAYERS,
                 dropout_rate: float = DROPOUT, output_steps: int = 1):
        # Model configuration
        self.input_size = int(input_size)
        self.model_name = model_name.lower()
        self.layers = layers or LAYERS
        self.dropout_rate = float(dropout_rate)
        self.output_steps = int(output_steps)  # For multistep prediction
        
        # Determine if bidirectional
        self.bidirectional = (self.model_name == 'bilstm')
        
        # Build the model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the modelarchitecture."""
        model = Sequential(name=self.model_name)
        # Input layer flexible (n_1, n_2, features)
        from tensorflow.keras import Input  # type: ignore
        model.add(Input(shape=(None, self.input_size)))
        
        # Add RNN layers
        for i, units in enumerate(self.layers):
            return_sequences = i < len(self.layers) - 1
            
            if self.model_name == 'bilstm':
                # Bidirectional LSTM
                model.add(Bidirectional(
                    LSTM(units=units, return_sequences=return_sequences)
                ))
            else:
                # Regular RNN layers
                Layer = _LAYER[self.model_name]
                model.add(Layer(
                    units=units, 
                    return_sequences=return_sequences
                ))
            
            # Add dropout after each layer for regularisation, option 2: + dropout and return_sequences
            if self.dropout_rate > 0 and return_sequences:
                model.add(Dropout(self.dropout_rate))
        
        # Final prediction layer
        # When output_steps=1: units=1 (single prediction)
        # When output_steps>1: units=output_steps (multiple predictions)
        model.add(Dense(units=self.output_steps, activation='linear'))
        
        return model

    def fit(self, x_train, y_train, epochs=25, batch_size=32, learning_rate=1e-3, optimizer: str = 'adam'):
        """Train a TensorFlow model using built-in Keras features."""
        # Compile model
        optimizers = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}
        opt_cls = optimizers[optimizer.lower()]
        opt = opt_cls(learning_rate=learning_rate)
        self.model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
        
        # Train and capture history
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Store history for plotting
        self.training_history = history

        return self

    def predict_and_evaluate(self, x_test, y_test):
        """Make predictions with the model and return comprehensive metrics."""
        predictions = self.model(x_test)
        # Use Keras 3+ API: compute_loss(x, y, y_pred, sample_weight=None, training=False)
        loss = self.model.compute_loss(x_test, y_test, predictions, sample_weight=None, training=False)
        
        # Calculate additional metrics manually
        import numpy as np
        
        # Convert to numpy for calculations to match y_test
        predictions_np = predictions.numpy()
        loss_np = loss.numpy()
        
        # Calculate MAE and RMSE
        if self.output_steps > 1:
            # Multistep prediction: calculate metrics for each step
            # Handle shape mismatch: y_test might be (n_samples, steps, 1) while predictions_np is (n_samples, steps)
            if len(y_test.shape) == 3 and y_test.shape[2] == 1:
                y_test_flat = y_test.reshape(y_test.shape[0], y_test.shape[1])
            else:
                y_test_flat = y_test
                
            mae_per_step = np.mean(np.abs(y_test_flat - predictions_np), axis=0)
            rmse_per_step = np.sqrt(np.mean((y_test_flat - predictions_np) ** 2, axis=0))
            overall_mae = np.mean(mae_per_step)
            overall_rmse = np.sqrt(np.mean((y_test_flat - predictions_np) ** 2))
            
            metrics = {
                'loss': loss_np,
                'mae': overall_mae,
                'rmse': overall_rmse,
                'mae_per_step': mae_per_step,
                'rmse_per_step': rmse_per_step,
                'loss_name': 'mean_squared_error',
                'mae_name': 'mean_absolute_error',
                'rmse_name': 'root_mean_squared_error'
            }
        else:
            # Single-step prediction: original logic
            mae = np.mean(np.abs(y_test - predictions_np))
            rmse = np.sqrt(np.mean((y_test - predictions_np) ** 2))
            
            metrics = {
                'loss': loss_np,
                'mae': mae,
                'rmse': rmse,
                'loss_name': 'mean_squared_error',
                'mae_name': 'mean_absolute_error',
                'rmse_name': 'root_mean_squared_error'
            }
        
        return predictions_np, metrics

    def save_model(self, filepath):
        """Save model to file."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model from saved state."""
        keras_model = tf.keras.models.load_model(filepath)  # type: ignore
        self.model = keras_model
        print(f"Model loaded from {filepath}")
        return self
