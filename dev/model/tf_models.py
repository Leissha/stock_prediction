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
                 dropout_rate: float = DROPOUT):
        # Model configuration
        self.input_size = int(input_size)
        self.model_name = model_name.lower()
        self.layers = layers or LAYERS
        self.dropout_rate = float(dropout_rate)
        
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
        model.add(Dense(units=1, activation='linear'))
        
        return model

    def fit(self, x_train, y_train, epochs=25, batch_size=32, learning_rate=1e-3, optimizer: str = 'adam'):
        """Train a TensorFlow model using built-in Keras features."""
        # Compile model
        optimizers = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}
        opt_cls = optimizers[optimizer.lower()]
        opt = opt_cls(learning_rate=learning_rate)
        self.model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
        
        # Train
        self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
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
        
        # Calculate MAE manually
        mae = np.mean(np.abs(y_test - predictions_np))
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_test - predictions_np) ** 2))
        
        # Create metrics dictionary
        metrics = {
            'loss': loss_np,
            'mae': mae,
            'rmse': rmse,
            'loss_name': 'mean_squared_error',  # From model compilation
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
