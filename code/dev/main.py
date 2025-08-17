# File: stock_prediction.py (Enhanced Version)
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

import numpy as np
import pandas as pd
import os
from loguru import logger
from model import LSTMModel
from data_preprocessing.data_loading import load_stock_data
from utils.file_handling import ensure_directory_exists, check_file_existence
from config.data import *
from utils.plots import plot_predictions
from data_input_prep import prepare_data_input

#------------------------------------------------------------------------------
# TODO 3: Build and save model
#------------------------------------------------------------------------------
def build_model(x_train, y_train):
    """
    Build and train the model with caching
    """
    model_file = os.path.join('models/', 'stock_prediction_model.h5')
    
    # Check if model exists
    model = check_file_existence(model_file)
    if model is None:
        logger.info("Building new model")
        model = LSTMModel(x_train, y_train, model_name="stock_prediction_model")
        model.train_and_save_model() # Train the model
    return model


#------------------------------------------------------------------------------
# Main execution
#------------------------------------------------------------------------------
def main():
    for dir in ["models", "cache", "results"]:
        ensure_directory_exists(f"dev/{dir}")
    
    # 1. Load processed data
    logger.info("=== Loading and Processing Data ===")
    raw_data = load_stock_data(COMPANY, TRAIN_START, TRAIN_END)
    prepared_data = prepare_data_input(raw_data, price_value=PRICE_VALUE, prediction_days=PREDICTION_DAYS)
    
    logger.info(f"Training data shape: x_train={prepared_data['x_train'].shape}, y_train={prepared_data['y_train'].shape}")
    
    # 2. Extract the prepared data
    x_train = prepared_data['x_train']
    y_train = prepared_data['y_train']
    x_test = prepared_data['x_test']
    y_test = prepared_data['y_test']
    scaler = prepared_data['scaler']
    
    # TODO 3: Build and save model
    logger.info("=== Building Model ===")
    model = build_model(x_train, y_train)
    
    # TODO 4: Better test data processing
    logger.info("=== Processing Test Data ===")
    test_data = load_stock_data(COMPANY, TEST_START, TEST_END)
    
    # Use the same scaler for test data
    actual_prices = test_data[PRICE_VALUE].values
    total_dataset = pd.concat((raw_data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    
    # Make predictions
    x_test = []
    for x in range(PREDICTION_DAYS, len(model_inputs)):
        x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    # Plot results
    plot_predictions(actual_prices, predicted_prices, f"dev/results/{COMPANY}_{PRICE_VALUE}_{PREDICTION_DAYS}_predictions.png")
    
    # Predict next day
    real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    logger.info(f"Next day prediction: {prediction[0][0]:.2f}")

if __name__ == "__main__":
    main()
