from abc import ABC, abstractmethod
import os

class BaseModel(ABC):
    def __init__(self, x_train, y_train, model_name):
        self.x_train = x_train
        self.y_train = y_train
        self.model_name = model_name
        self.model_file = os.path.join("models", self.model_name + ".h5")

    @abstractmethod
    def build_model(self):
        pass
    
    @abstractmethod
    def train_and_save_model(self):
        pass
    
    @abstractmethod
    def predict(self, x_test):
        pass

