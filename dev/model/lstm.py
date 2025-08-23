from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dropout, Dense # type: ignore
from .base_model import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.file_handling import check_file_existence
from loguru import logger

class LSTMModel(BaseModel):
    def __init__(self, model_name="lstm", model=None):
        super().__init__(model_name=model_name, model=model)

    def create_model(self, x_train):
        """
        Build the LSTM model
        """
        # Check if model exists and load it
        model_architecture = self.model_path + ".h5"
        self.model = check_file_existence(model_architecture)
        if self.model is not None:
            logger.info(f"Loading existing model from {model_architecture}")
            return self.model
        
        # Build new model
        logger.info("Building new LSTM model")
        self.model = Sequential()
        
        # First LSTM layer with input shape
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        self.model.add(Dropout(0.2))
        
        # Second LSTM layer
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        
        # Third LSTM layer
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        
        # Output layer
        self.model.add(Dense(units=1))
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        return self.model
        