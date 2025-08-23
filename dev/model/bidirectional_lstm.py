from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional, Input # type: ignore
from .base_model import BaseModel

class BidirectionalLSTMModel(BaseModel):
    def __init__(self, model_name="bidirectional_lstm", model=None):
        super().__init__(model_name=model_name, model=model)
    
    def create_model(self, sequence_length, n_features, units=50, cell=LSTM, n_layers=2, dropout=0.2,
                loss="mean_squared_error", optimizer="adam", bidirectional=True):

        self.model = Sequential()
        
        for i in range(n_layers):
            if i == 0:
                # first layer
                if bidirectional:
                    self.model.add(Bidirectional(cell(units, return_sequences=True), 
                                               input_shape=(sequence_length, n_features)))        
                else:
                    self.model.add(cell(units, return_sequences=True, 
                                      input_shape=(sequence_length, n_features)))
            elif i == n_layers - 1:
                # last layer
                if bidirectional:
                    self.model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    self.model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                if bidirectional:
                    self.model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    self.model.add(cell(units, return_sequences=True))
            # add dropout after each layer
            self.model.add(Dropout(dropout))
        
        self.model.add(Dense(1))
        self.model.compile(loss=loss, optimizer=optimizer)
        return self.model
        

