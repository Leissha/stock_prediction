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
from model import *
from data_preprocessing.data_processor import DataProcessor
from utils.file_handling import check_file_existence, save_data
from config.data import *
from utils.plots import plot_predictions
from argparse import ArgumentParser
from model.lstm import LSTMModel
from model.bidirectional_lstm import BidirectionalLSTMModel

#------------------------------------------------------------------------------
# Train model
#------------------------------------------------------------------------------
def train(base_path) -> None:
    data = check_file_existence(f"dev/cache/processed_data/{base_path}.pkl")
    if data is None:
        # Create DataProcessor instance
        processor = DataProcessor(cache_dir='dev/cache')
        
        # Use the comprehensive data processing method
        data = processor.data_processing(
            start_date=START_DATE,
            end_date=END_DATE,
            ticker=TICKER,
            n_steps=PREDICTION_DAYS,
            lookup_step=1,
            shuffle=SHUFFLE,
            splitting_method=SPLIT_METHOD,
            test_size=TEST_SIZE,
            feature_columns=FEATURES,
            target_feature='volume',  # Predict Volume instead of Close price
            scale=SCALE,
            base_path=base_path
        )
        
        if data:
            save_data(data, f"dev/cache/processed_data/{base_path}.pkl")
            # Also save scalers separately for easy access
            save_data(data['column_scaler'], f"dev/cache/processed_data/{base_path}_scalers.pkl")
        else: 
            logger.error("Data processing failed")
            exit(1)
    
    # Extract the processed data (scaled)
    x_train = data['X_train']
    y_train = data['y_train']
    x_test = data['X_test']
    y_test = data['y_test']
    
    logger.info(f"Training data shape: x_train={x_train.shape}, y_train={y_train.shape}")
    logger.info(f"Test data shape: x_test={x_test.shape}, y_test={y_test.shape}")

    model_path = f"dev/cache/trained_models/{base_path}.h5"
    model = check_file_existence(model_path)
    if model is None: 
        # 2. Build and train model
        logger.info("=== Building Model ===")
        if MODEL_NAME.lower() == "lstm":
            model = LSTMModel(model_name=MODEL_NAME, model=None)
            model.create_model(x_train)
        elif MODEL_NAME.lower() == "bidirectional_lstm":
            model = BidirectionalLSTMModel(model_name=MODEL_NAME, model=None)
            model.create_model(sequence_length=x_train.shape[1], n_features=x_train.shape[2]) 
        else:
            logger.error(f"Invalid model name: {MODEL_NAME}. Please re-enter: lstm or bidirectional_lstm.")
            exit(1)
    
        # Train the model
        logger.info("=== Training Model ===")
        model.train(x_train, y_train, epochs=25, batch_size=32, validation_split=0.1)
        
        # Save the trained model
        logger.info("=== Saving Model ===")
        model.save_model(model_path)
        logger.info(f"Model saved to: {model_path}")
    
    # 3. Make predictions on test data
    logger.info("=== Making Predictions ===")
    
def predict(base_path) -> None:
    data = check_file_existence(f"dev/cache/processed_data/{base_path}.pkl")
    if data is None:
        logger.error("Data not found")
        exit(1)
    
    x_test = data['X_test']
    scalers = data['column_scaler']
    FEATURES = data['feature_columns']
    y_test = data['y_test']
    
    model = check_file_existence(f"dev/cache/trained_models/{base_path}.h5")
    if model is None:
        logger.error("Model not found")
        exit(1)
    
    predicted_prices = model.predict(x_test)
    
    # Model predicts in scaled space, so we need to inverse transform predictions
    # Both predictions and y_test are scaled and need inverse transform
    if predicted_prices is not None and 'future' in scalers:
        # Inverse transform predictions from scaled to original scale
        predicted_prices = scalers['future'].inverse_transform(predicted_prices.reshape(-1, 1)).flatten()
        # Also inverse transform y_test since it's scaled too
        actual_prices = scalers['future'].inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        actual_prices = y_test
        if predicted_prices is None:
            logger.error("Model prediction failed")
            return
    
    # 4. Plot results
    plot_path = f"dev/results/{base_path}.png"
    # Get test dates from the data - use actual dates from test_df index
    test_df = data.get('test_df', pd.DataFrame())
    test_dates = test_df.index
    
    logger.info(f"Test dates info: len={len(test_dates)}, actual_prices len={len(actual_prices)}")
    logger.info(f"First few test dates: {test_dates[:5].tolist() if len(test_dates) > 0 else 'No dates'}")
    
    if len(test_dates) == 0 or len(test_dates) != len(actual_prices):
        # Fallback to range if test_df not available or length mismatch
        logger.info("Using range fallback for x-axis")
        test_dates = range(len(actual_prices))
        plot_predictions(actual_prices, predicted_prices, test_dates, plot_path)
    else:
        # Create DataFrame to sort by date for clean visualization
        plot_df = pd.DataFrame({
            'date': test_dates,
            'actual': actual_prices,
            'predicted': predicted_prices
        }).sort_values('date')
        
        logger.info(f"Sorted date range: {plot_df['date'].min()} to {plot_df['date'].max()}")
        plot_predictions(plot_df['actual'].values, plot_df['predicted'].values, plot_df['date'].values, plot_path)
    
    # 5. Predict next day using last sequence
    logger.info("=== Next Day Prediction ===")
    last_sequence = data['last_sequence']
    if last_sequence is not None:
        # Reshape for prediction (add batch dimension)
        last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
        
        prediction = model.predict(last_sequence)
        if prediction is not None and 'future' in scalers:
            # Inverse transform prediction from scaled to original scale
            prediction = scalers['future'].inverse_transform(prediction.reshape(-1, 1))
            logger.info(f"Next day prediction: {prediction[0][0]:.2f}")
        else:
            logger.error("Next day prediction failed")
    else:
        logger.error("Last sequence not available for prediction")
    
    # 6. Print some evaluation metrics
    logger.info("=== Model Evaluation ===")
    if 'future' in scalers:
        sc = scalers['future']
        logger.info(f"Scaler min/max for future: {sc.data_min_[0]}, {sc.data_max_[0]}")
        logger.info(f"predicted (original) range: {predicted_prices.min():.4f}-{predicted_prices.max():.4f}")
        logger.info(f"actual (original) range: {actual_prices.min():.4f}-{actual_prices.max():.4f}")
    
    # 7. Calculate trading-based accuracy metrics (same as v0.1 and p1)
    logger.info("=== Trading Accuracy Metrics ===")
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.evaluating_utils import calculate_trading_metrics
        
        # Get current prices for trading accuracy calculation
        current_prices = data.get('current_prices', actual_prices)  # Fallback to actual_prices if not available
        
        # Get next day prediction for CSV output
        next_day_predicted = prediction[0][0] if prediction is not None else 0
        
        # Save to CSV
        csv_filename = f"dev/results/{base_path}.csv"
        calculate_trading_metrics(
            actual_prices=actual_prices,
            predicted_prices=predicted_prices,
            current_prices=current_prices,
            lookup_step=1,
            future_price=next_day_predicted,
            loss_value=0,
            loss_name="loss",
            filename=csv_filename,
            scale=SCALE,
            feature_names=FEATURES,
            evaluation_mode="price"
        )
        logger.info(f"Accuracy metrics saved to: {csv_filename}")
        
    except ImportError as e:
        logger.warning(f"Could not import accuracy_utils: {e}")
        logger.info("Using standard metrics only")

#------------------------------------------------------------------------------
# Argument parsing
#------------------------------------------------------------------------------
def parse_args():
    """
    Parse command line arguments for the stock prediction model
    """
    parser = ArgumentParser(description='Stock Price Prediction with LSTM')
    # Data loading arguments
    parser.add_argument("--company", type=str, default=COMPANY, 
                       help="Company ticker symbol (default: %(default)s)")
    parser.add_argument("--start_date", type=str, default=TRAIN_START,
                       help="Start date for training data (default: %(default)s)")
    parser.add_argument("--end_date", type=str, default=TRAIN_END,
                       help="End date for training data (default: today)")
    
    # Feature selection arguments
    parser.add_argument("--features", nargs='+', 
                       default=[PRICE_VALUE],
                       help="Features to use for prediction (default: %(default)s)")
    parser.add_argument("--prediction_days", type=int, default=PREDICTION_DAYS,
                       help="Number of days to look back for prediction (default: %(default)s)")
    
    # Data processing arguments
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Test set size ratio (default: %(default)s)")
    parser.add_argument("--split_method", type=str, default='date', choices=['date', 'ratio', 'random'],
                       help="Data splitting method (default: %(default)s)")
    parser.add_argument("--shuffle", action='store_true', default=True,
                       help="Shuffle the data (default: True)")
    parser.add_argument("--scale", action='store_true', default=True,
                       help="Scale the features (default: True)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="Current available models: lstm, bidirectional_lstm", choices=['lstm', 'bidirectional_lstm'])
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    TICKER = args.company
    START_DATE = args.start_date
    END_DATE = args.end_date
    FEATURES = args.features
    PREDICTION_DAYS = args.prediction_days
    TEST_SIZE = args.test_size
    SPLIT_METHOD = args.split_method
    SHUFFLE = args.shuffle
    SCALE = args.scale
    MODEL_NAME = args.model_name
    
    base_path = f"{START_DATE}_{COMPANY}_{FEATURES}_seq-{PREDICTION_DAYS}-step_1_{MODEL_NAME}"
    
    # Create necessary directories
    for dir in ["cache", "cache/trained_models", "cache/processed_data", "results"]:
        os.makedirs(f"dev/{dir}", exist_ok=True)
        
    train(base_path)
    predict(base_path)
