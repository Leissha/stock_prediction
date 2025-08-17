from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from loguru import logger
from model.base_model import BaseModel
from utils.file_handling import check_file_existence

class LSTMModel(BaseModel):
    def __init__(self, x_train, y_train, model_name):
        super().__init__(x_train, y_train, model_name = "lstm")

    def build_model(self):
        """
        Build and train the model with caching
        """
        model = check_file_existence(self.model_file)
        if model is None:
            model = Sequential()
            
        model = Sequential()
            
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
    def train_and_save_model(self):
        """
        Train the model
        """
        model = self.build_model()
        model.fit(self.x_train, self.y_train, epochs=25, batch_size=32, checkpoint_path=self.model_file)
        model.save(self.model_file)
        logger.info(f"Model saved to {self.model_file}")

    def predict(self, x_test):
        """
        Predict the model
        """ 
        model = self.build_model()
        model.predict(x_test)
        return model

