"""
TensorFlow Model for LSTM, GRU, RNN, BiLSTM.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from loguru import logger
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional  # type: ignore
from typing import List, Optional
import numpy as np
from config.data import MODEL_NAME, LAYERS, DROPOUT
from utils.plots import plot_training_metrics

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

    def fit(self, x_train, y_train, epochs=25, batch_size=32, learning_rate=1e-3, optimizer: str = 'adam', validation_data=None, meta_path=None):
        """Train a TensorFlow model using built-in Keras features.
        validation_data: optional tuple (X_val, y_val) to monitor validation loss.
        """
        # Compile model
        optimizers = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}
        opt_cls = optimizers[optimizer.lower()]
        opt = opt_cls(learning_rate=learning_rate)
        self.model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
        
        # Train and capture history
        fit_kwargs = {
            'x': x_train,
            'y': y_train,
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': 1,
        }
        
        if validation_data is not None:
            fit_kwargs['validation_data'] = validation_data
        # Early stopping if validation is provided
        callbacks = []
        if validation_data is not None:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, mode='min'))

        history = self.model.fit(**fit_kwargs, callbacks=callbacks)

        plot_training_metrics(history, save_path=f"cache/trained_models/{meta_path}_training.png")
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions for test data.

        Args:
            x_test: Input sequences shape (N, L, F)

        Returns:
            predictions: Shape (N, K) where K = output_steps
        """
        predictions = self.model(x_test)
        predictions_np = predictions.numpy()
        if predictions_np.ndim == 1:
            predictions_np = predictions_np[:, None]
        return predictions_np

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
