from abc import ABC, abstractmethod
import os

from loguru import logger

class BaseModel(ABC):
    def __init__(self, model_name="base_model", model=None):
        self.model = model
        self.model_name = model_name
        # Save path like dev/trained_models/<model_name>.h5
        self.model_path = os.path.join("dev", "trained_models", f"{self.model_name}")

    @abstractmethod
    def create_model(self):
        pass
    
    def train(self, x_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1):
        """
        Train the model
        """
        if self.model is None:
            logger.error("Model is not created. Please create the model first.")
            return None
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                               validation_split=validation_split, verbose=verbose)
        return history
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.model is None:
            logger.error("Model is not created. Please create the model first.")
            return None
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None
    
    def predict(self, x_test):
        """
        Predict using the model
        """ 
        if self.model is None:
            logger.error("Model is not created. Please create the model first.")
            return exit(1)
        try:
            predictions = self.model.predict(x_test)
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
