"""
TensorFlow Model for LSTM, GRU, RNN, BiLSTM + CNN hybrid & Attention models
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from loguru import logger
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    LSTM,
    GRU,
    SimpleRNN,
    Dense,
    Dropout,
    Bidirectional,
    Conv1D,
    MultiHeadAttention,
    LayerNormalization,
    Concatenate,
    GlobalAveragePooling1D,
)  # type: ignore
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
                 dropout_rate: float = DROPOUT, output_steps: int = 1, task_type: str = "regression",
                 attn_heads: int = 4, attn_key_dim: int = 16):
        """
        Initialize the TFModel.

        Args:
            input_size: Number of input features.
            model_name: Name of the model architecture (e.g., 'lstm', 'gru').
            layers: List of layer sizes.
            dropout_rate: Dropout rate for regularization.
            output_steps: Number of output steps (for multi-step forecasting).
            task_type: Task type ('binary', 'regression', or 'multi_step').
        """
        # Model configuration
        self.input_size = int(input_size)
        self.model_name = model_name.lower()
        self.layers = layers or LAYERS
        self.dropout_rate = float(dropout_rate)
        self.output_steps = int(output_steps)  # For multistep prediction
        # Task type: 'regression' | 'binary'
        self.task_type = task_type.lower()
        # Attention params (used only by attention_* variants)
        self.attn_heads = int(attn_heads)
        self.attn_key_dim = int(attn_key_dim)
        
        # Determine if bidirectional
        self.bidirectional = (self.model_name == 'bilstm')
        
        # Build the model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the model architecture."""
        model = Sequential(name=self.model_name)
        # Input layer flexible (n_1, n_2, features)
        from tensorflow.keras import Input  # type: ignore
        model.add(Input(shape=(None, self.input_size)))

        # Route to appropriate builder based on model type
        if self.model_name in ['cnn_lstm', 'cnn_rnn', 'cnn_gru', 'cnn_bilstm']:
            self._build_cnn_hybrid_layers(model)
        elif self.model_name in ['attention_lstm', 'attention_gru', 'attention_rnn', 'attention_bilstm']:
            return self._build_attention_layers()
        else:
            self._build_rnn_layers(model)

        # Add final prediction layer
        self._add_output_layer(model)

        return model

    def _build_cnn_hybrid_layers(self, model):
        """Build CNN+RNN hybrid architecture
        - CNN layers for spatial feature extraction
        - RNN layers for sequential modeling
        - Dense layer for classification
        """
        # CNN layers
        model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv1'))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='conv2'))
        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate, name='dropout_cnn'))

        # RNN layers - choose variant based on model name
        if self.model_name == 'cnn_gru':
            model.add(GRU(64, return_sequences=True, name='gru1'))
            model.add(GRU(32, return_sequences=False, name='gru2'))
        elif self.model_name == 'cnn_bilstm':
            model.add(Bidirectional(LSTM(64, return_sequences=True), name='bilstm1'))
            model.add(Bidirectional(LSTM(32, return_sequences=False), name='bilstm2'))
        else:  # cnn_lstm / cnn_rnn default to LSTM
            model.add(LSTM(64, return_sequences=True, name='lstm1'))
            model.add(LSTM(32, return_sequences=False, name='lstm2'))

        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate, name='dropout_rnn'))

        # Dense layer
        model.add(Dense(32, activation='relu', name='dense1'))
        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate, name='dropout_dense'))

    def _build_rnn_layers(self, model):
        """Build standard RNN/LSTM/GRU/BiLSTM architecture layers."""
        for i, units in enumerate(self.layers):
            return_sequences = i < len(self.layers) - 1

            if self.model_name == 'bilstm':
                # Bidirectional LSTM
                model.add(Bidirectional(
                    LSTM(units=units, return_sequences=return_sequences)
                ))
            else:
                # Regular RNN layers (LSTM, GRU, SimpleRNN)
                Layer = _LAYER[self.model_name]
                model.add(Layer(
                    units=units,
                    return_sequences=return_sequences
                ))

            # Add dropout after each sequence-returning layer
            if self.dropout_rate > 0 and return_sequences:
                model.add(Dropout(self.dropout_rate))

    def _build_attention_layers(self):
        """
        RNN with multi-head attention mechanism.

        The attention mechanism allows the model to focus on the most relevant
        time steps when making predictions, rather than treating all time steps equally.

        Architecture:
        1. RNN layer processes the sequence
        2. Multi-head attention attends to important time steps
        3. Concatenate RNN output with attention output
        4. Dense layers for classification

        Why this might help:
        - Stock movements may depend more on recent events (earnings) than older data
        - Attention can learn which time steps matter most
        - Multi-head attention captures different types of patterns
        """
        from tensorflow.keras import Input  # type: ignore
        inputs = Input(shape=(None, self.input_size), name='input')

        # RNN encoder with sequences
        if self.model_name == 'attention_bilstm':
            rnn_out = Bidirectional(LSTM(64, return_sequences=True), name='rnn')(inputs)
        elif self.model_name == 'attention_gru':
            rnn_out = GRU(64, return_sequences=True, name='rnn')(inputs)
        elif self.model_name == 'attention_rnn':
            rnn_out = SimpleRNN(64, return_sequences=True, name='rnn')(inputs)
        else:  # attention_lstm
            rnn_out = LSTM(64, return_sequences=True, name='rnn')(inputs)

        if self.dropout_rate > 0:
            rnn_out = Dropout(self.dropout_rate, name='dropout_rnn')(rnn_out)

        # Self-attention over time
        # Multi-head attention
        # Query, Key, Value all come from RNN output
        attn = MultiHeadAttention(num_heads=self.attn_heads, key_dim=self.attn_key_dim, name='self_attn')(rnn_out, rnn_out)
        # Layer normalization
        attn = LayerNormalization(name='attn_ln')(attn)

        # # Take the last time step from attention output + pooled attention context
        last_step = rnn_out[:, -1, :]
        context = GlobalAveragePooling1D(name='attn_pool')(attn)
        
        # Concatenate RNN and attention outputs
        merged = Concatenate(name='concat')([last_step, context])

        x = Dense(32, activation='relu', name='dense1')(merged)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate, name='dropout_dense')(x)

        if self.task_type == 'binary':
            outputs = Dense(1, activation='sigmoid', name='output')(x)
        else:
            outputs = Dense(self.output_steps, activation='linear', name='output')(x)

        from tensorflow.keras.models import Model  # type: ignore
        return Model(inputs=inputs, outputs=outputs, name=self.model_name)

    def _add_output_layer(self, model):
        """Add final prediction layer based on task type."""
        if self.task_type == 'binary':
            # Binary classification: sigmoid activation
            model.add(Dense(units=1, activation='sigmoid', name='output'))
        else:
            # Regression: linear activation
            # output_steps=1: single prediction, output_steps>1: multiple predictions
            model.add(Dense(units=self.output_steps, activation='linear', name='output'))

    def fit(self, x_train, y_train, epochs=25, batch_size=32, learning_rate=1e-3, optimizer: str = 'huber', validation_data=None, meta_path=None, class_weight=None, patience=10):
        """
        Train TensorFlow model:
        validation_data: optional tuple (X_val, y_val) to monitor validation loss.
        patience: Patience for ReduceLROnPlateau callback (default: 10)
        """
        # Compile model
        optimizers = {'adam': Adam, 'rmsprop': RMSprop, 'sgd': SGD}
        opt_cls = optimizers[optimizer.lower()]
        opt = opt_cls(learning_rate=learning_rate)
        if self.task_type == 'binary':
            self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            # Prefer MAE to prevent collapsing to near-zero mean under MSE
            self.model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mae'])
        
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
            # Early stopping and LR scheduling to stabilize validation
            # Increased patience to allow more exploration before stopping
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=20,  # Increased from 15 to allow more training
                restore_best_weights=True,
                mode='min',
                min_delta=0.0001  # Reduced from 0.001 for finer sensitivity
            ))
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=patience, min_lr=1e-6, verbose=1  # Use configurable patience
            ))

        # Initialize output bias to training mean to avoid zero-return collapse (regression only)
        if self.task_type != 'binary':
            try:
                y_mean = np.mean(y_train, axis=0)
                if np.ndim(y_mean) == 0:
                    y_mean = np.array([float(y_mean)], dtype=np.float32)
                out = self.model.get_layer('output')
                w = out.get_weights()
                if len(w) == 2 and w[1].shape[0] == y_mean.shape[-1]:
                    w[1] = np.asarray(y_mean, dtype=w[1].dtype)
                    out.set_weights(w)
            except Exception:
                pass

        history = self.model.fit(**fit_kwargs, callbacks=callbacks, class_weight=class_weight)

        # Store actual epochs trained (accounting for early stopping)
        self.epochs_trained = len(history.history['loss'])
        logger.info(f"Training completed: {self.epochs_trained}/{epochs} epochs")

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
        # CRITICAL: Clean input before prediction to prevent NaN propagation
        x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)

        predictions = self.model(x_test)
        predictions_np = predictions.numpy()
        if predictions_np.ndim == 1:
            predictions_np = predictions_np[:, None]

        # Safety: Clean output as well
        predictions_np = np.nan_to_num(predictions_np, nan=0.0, posinf=0.0, neginf=0.0)
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
