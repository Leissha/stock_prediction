# File: train.py
# Authors: Kha Anh Nguyen :)
# Date: 24/08/2025

# Code modified from:
# File: stock_prediction.py (Enhanced Version)
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# and
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

import numpy as np
import pandas as pd
import os
from loguru import logger
from data_preprocessing.data_processor import DataProcessor
from utils.file_handling import check_file_existence, save_data
from config.data import *
from utils.plots import plot_predictions
from argparse import ArgumentParser
from model.lstm import LSTMModel
from model.bidirectional_lstm import BidirectionalLSTMModel
from utils.evaluating_utils import calculate_trading_metrics    

def predict_and_transform(model, x_test, y_test, target_scaler=None):
    """
    Simplified prediction function with single-scaler inverse transform
    
    Args:
        model: Trained model
        x_test: Test input data
        y_test: Test target data (scaled)
        data: Dictionary containing scalers and metadata from DataProcessor

    Returns:
        tuple: (actual_prices, predicted_prices, loss_val) - all in original scale
    """
    # Make predictions using model
    predictions = model.predict(x_test)
    
    # Evaluate model performance on scaled data
    loss_val = model.evaluate(x_test, y_test, verbose=0)
    
    # Inverse transform using separate scaler for target feature
    if target_scaler is not None:
        actual_prices = target_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).reshape(-1)
        predicted_prices = target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(-1)
        
        logger.info(f"Inverse transformed using scaler: {target_scaler}")
    else:
        # No scaling was applied, use values as-is
        actual_prices = y_test.reshape(-1)
        predicted_prices = predictions.reshape(-1)
        logger.info("No inverse transform applied - values used as-is")
        
    return actual_prices, predicted_prices, loss_val

#------------------------------------------------------------------------------
# Train Model
#------------------------------------------------------------------------------
def train(base_path) -> None:
    """Train the LSTM model on stock data"""
    logger.info("=== TRAINING PHASE ===")
    
    # Check if data already processed
    data = check_file_existence(f"dev/cache/processed_data/{base_path}.pkl")
    if data is None:
        # Create DataProcessor instance
        processor = DataProcessor(cache_dir='dev/cache')
        
        # Use the comprehensive data processing method
        data = processor.data_processing(
            start_date=START_DATE,
            end_date=END_DATE,
            ticker=TICKER,				    # Stock company 
            lag_days=LAG_DAYS,				# learning window size 
            lookup_step=1,				    # prediction horizon
            shuffle=SHUFFLE,				# shuffle training set or not
            splitting_method=SPLIT_METHOD, 	# ratio /date /random split
            test_size=TEST_SIZE,
            target_feature='Close',			# Predict Close price
            scale=SCALE
        )
        
        if data:
            save_data(data, f"dev/cache/processed_data/{base_path}.pkl")
        else: 
            logger.error("Data processing failed")
            exit(1)
    
    # Extract the processed data
    x_train = data['X_train']
    y_train = data['y_train']
    x_test = data['X_test']
    y_test = data['y_test']
    
    logger.info(f"Training data shape: x_train={x_train.shape}, y_train={y_train.shape}")
    logger.info(f"Test data shape: x_test={x_test.shape}, y_test={y_test.shape}")

    # Check if model already trained
    model_path = f"dev/cache/trained_models/{base_path}.h5"
    model = check_file_existence(model_path)
    if model is None: 
        logger.info("Building and training model...")
        # Build model based on specified type
        if MODEL_NAME.lower() == "lstm":
            model = LSTMModel(model_name=MODEL_NAME, model=None)
        elif MODEL_NAME.lower() == "bidirectional_lstm":
            model = BidirectionalLSTMModel(model_name=MODEL_NAME, model=None)
        else:
            logger.error(f"Invalid model name: {MODEL_NAME}. Please re-enter: lstm or bidirectional_lstm.")
            exit(1)
        
        # Create the model architecture
        model.create_model(x_train)
        
        # Train the model
        model.train(x_train, y_train, epochs=25, batch_size=32, validation_split=0.1)
        
        # Save the trained model
        model.save_model(model_path)
        logger.info(f"Model saved to: {model_path}")
    else:
        logger.info("Using existing trained model")

    logger.info("=== TESTING PHASE ===")
    
    target_feature = data['target_feature'].lower()
    target_key = f"{TICKER}_{target_feature}"
    target_scaler = data['scalers'].get(target_key)
    
    # Make predictions on test set using simplified function
    actual_prices, predicted_prices, loss_val = predict_and_transform(model, x_test, y_test, target_scaler)
    
    # Generate plots
    plot_path = f"dev/results/{base_path}.png"
    test_df = data.get('test_df', pd.DataFrame())
    test_dates = test_df.index
    
    plot_predictions(actual_prices, predicted_prices, plot_path, dates=test_dates)

    # Calculate metrics and save to CSV
    csv_filename = f"dev/results/{base_path}.csv"
    # Get the latest prediction as a plain float for reporting
    pred_arr = np.asarray(predicted_prices).reshape(-1)
    future_price = float(pred_arr[-1]) if pred_arr.size > 0 else 0.0
    
    # Get current prices (previous day's actual prices for realistic trading)
    current_prices = []
    for i in range(len(actual_prices)):
        if i == 0:
            # For first prediction, use the last training price
            # We need to get this from the original test_df before scaling
            current_prices.append(actual_prices[i])  # Fallback to same price
        else:
            current_prices.append(actual_prices[i-1])
    
    metrics = calculate_trading_metrics(
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        current_prices=current_prices,
        lookup_step=1,
        future_price=future_price,
        loss_val=loss_val,
        filename=csv_filename,
        scale=SCALE,
        training_features=data.get('feature_columns', ['Close']),
        target_feature=data.get('target_feature', 'Close'),
    )
    
    logger.info(f"Test results saved to: {csv_filename}")
    return metrics


#------------------------------------------------------------------------------
# Command Line Arguments
#------------------------------------------------------------------------------
def parse_args():
    """Parse command line arguments for the stock prediction model"""
    parser = ArgumentParser(description='Stock Price Prediction with LSTM')
    
    # Data loading arguments
    parser.add_argument("--company", type=str, default=COMPANY,
                       help="Company ticker symbol (default: %(default)s)")
    parser.add_argument("--start_date", type=str, default=TRAIN_START,
                       help="Start date for data (default: %(default)s)")
    parser.add_argument("--end_date", type=str, default=TRAIN_END,
                       help="End date for data (default: %(default)s)")
    
    # Feature selection arguments (using all OHLCV features, predicting Close price)
    parser.add_argument("--target_feature", type=str, default=PRICE_VALUE,
                       help="Target feature to predict (default: Close)", choices=["Close", "Open", "High", "Low", "AdjClose", "Volume"])
    parser.add_argument("--lag_days", type=int, default=LAG_DAYS,
                       help="Number of days to look back for prediction (default: %(default)s)")
    
    # Data processing arguments
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Test size ratio (default: %(default)s)")
    parser.add_argument("--split_method", type=str, default="date",
                       choices=['date', 'random'],
                       help="Data splitting method (default: %(default)s)")
    parser.add_argument("--shuffle", action="store_true", default=False,
                       help="Shuffle data during training")
    parser.add_argument("--scale", action="store_true", default=True,
                       help="Scale the data")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="lstm",
                       help="Model type to use", 
                       choices=['lstm', 'bidirectional_lstm'])
    
    return parser.parse_args()

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    TICKER = args.company
    START_DATE = args.start_date
    END_DATE = args.end_date
    LAG_DAYS = args.lag_days
    TEST_SIZE = args.test_size
    SPLIT_METHOD = args.split_method
    SHUFFLE = args.shuffle
    SCALE = args.scale
    MODEL_NAME = args.model_name
    TARGET_FEATURE = args.target_feature    
    base_path = f"{START_DATE}_{TICKER}_{TARGET_FEATURE}_seq-{LAG_DAYS}-step_1_{MODEL_NAME}"
    
    # Create necessary directories
    for dir in ["cache", "cache/trained_models", "cache/processed_data", "cache/raw_data", "cache/scalers", "results"]:
        os.makedirs(f"dev/{dir}", exist_ok=True)
    
    # Execute the clean DRY pipeline
    print("="*60)
    print("STOCK PREDICTION PIPELINE")
    print("="*60)
    
    # 1. Train model
    train(base_path)
    
    print("="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)