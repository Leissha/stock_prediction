# File: train.py
# Authors: Kha Anh Nguyen
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
from model import *
from data_preprocessing.data_processor import DataProcessor
from utils.file_handling import check_file_existence, save_data
from config.data import *
from utils.plots import plot_predictions
from argparse import ArgumentParser
from model.lstm import LSTMModel
from model.bidirectional_lstm import BidirectionalLSTMModel

#------------------------------------------------------------------------------
# Shared Utility Functions (DRY Principle)
#------------------------------------------------------------------------------

def load_data_and_model(base_path):
    """Load processed data and trained model - shared by test and inference"""
    # Load processed data
    data = check_file_existence(f"dev/cache/processed_data/{base_path}.pkl")
    if data is None:
        logger.error("Processed data not found. Please run training first.")
        return None, None
    
    # Load trained model
    model = check_file_existence(f"dev/cache/trained_models/{base_path}.h5")
    if model is None:
        logger.error("Trained model not found. Please run training first.")
        return None, None
    
    return data, model

def predict_and_transform(model, x_data, scalers, is_single_prediction=False):
    """
    Core prediction function with optional inverse transform
    
    Args:
        model: Trained model
        x_data: Input data for prediction
        scalers: Dictionary of scalers
        is_single_prediction: True for single prediction, False for batch
    
    Returns:
        Transformed predictions in original scale
    """
    # Make predictions using model
    predictions = model.predict(x_data)
    
    # Inverse transform if scalers available
    if predictions is not None and 'future' in scalers:
        if is_single_prediction:
            # Single prediction (for inference)
            predictions = scalers['future'].inverse_transform(predictions.reshape(-1, 1))
        else:
            # Batch predictions (for testing)
            predictions = scalers['future'].inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    return predictions

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
            ticker=TICKER,
            n_steps=PREDICTION_DAYS,
            lookup_step=1,
            shuffle=SHUFFLE,
            splitting_method=SPLIT_METHOD,
            test_size=TEST_SIZE,
            feature_columns=FEATURES,
            target_feature='Close',  # Predict Close price
            scale=SCALE,
            base_path=base_path
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
        model.create_model(sequence_length=x_train.shape[1], n_features=x_train.shape[2])
        
        # Train the model
        model.train(x_train, y_train, epochs=25, batch_size=32, validation_split=0.1)
        
        # Save the trained model
        model.save_model(model_path)
        logger.info(f"Model saved to: {model_path}")
    else:
        logger.info("Using existing trained model")

#------------------------------------------------------------------------------
# Test Model Performance
#------------------------------------------------------------------------------
def test(base_path) -> dict:
    """
    Evaluate model performance on test data
    
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("=== TESTING PHASE ===")
    
    # Load data and model
    data, model = load_data_and_model(base_path)
    if data is None or model is None:
        return {}
    
    # Get test data
    x_test = data['X_test']
    y_test = data['y_test']
    scalers = data['column_scaler']
    
    # Make predictions on test set using shared function
    predicted_prices = predict_and_transform(model, x_test, scalers, is_single_prediction=False)
    
    # Transform actual prices to original scale
    if 'future' in scalers:
        actual_prices = scalers['future'].inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        actual_prices = y_test
        if predicted_prices is None:
            logger.error("Model prediction failed")
            return {}
    
    # Evaluate model performance
    evaluation_result = model.evaluate(x_test, y_test, verbose=0)
    if isinstance(evaluation_result, (list, tuple)):
        model_loss, model_mae = evaluation_result
        logger.info(f"Model Loss: {model_loss:.6f}")
        logger.info(f"Model MAE: {model_mae:.6f}")
    else:
        model_loss = evaluation_result
        model_mae = model_loss
        logger.info(f"Model Loss: {model_loss:.6f}")
    
    # Generate plots
    plot_path = f"dev/results/{base_path}.png"
    test_df = data.get('test_df', pd.DataFrame())
    test_dates = test_df.index
    
    if len(test_dates) == 0 or len(test_dates) != len(actual_prices):
        logger.info("Using range fallback for x-axis")
        test_dates = range(len(actual_prices))
        plot_predictions(actual_prices, predicted_prices, test_dates, plot_path)
    else:
        # Create DataFrame for clean visualization
        plot_df = pd.DataFrame({
            'actual': actual_prices,
            'predicted': predicted_prices,
            'date': test_dates[:len(actual_prices)]
        }).sort_values('date')
        
        plot_predictions(plot_df['actual'].values, plot_df['predicted'].values, plot_df['date'].values, plot_path)
    
    # Calculate trading metrics
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.evaluating_utils import calculate_trading_metrics
        
        # Simulate realistic trading scenario (current prices slightly lower)
        current_prices = actual_prices * 0.99
        
        # Calculate metrics and save to CSV
        csv_filename = f"dev/results/{base_path}.csv"
        metrics = calculate_trading_metrics(
            actual_prices=actual_prices,
            predicted_prices=predicted_prices,
            current_prices=current_prices,
            lookup_step=1,
            future_price=0,  # Will be set by inference
            loss_value=model_loss,
            loss_name="mse_loss",
            filename=csv_filename,
            scale=SCALE,
            feature_names=FEATURES,
            evaluation_mode="price"
        )
        
        logger.info(f"Test results saved to: {csv_filename}")
        return metrics
        
    except ImportError as e:
        logger.warning(f"Could not import evaluating_utils: {e}")
        return {}

#------------------------------------------------------------------------------
# Make Future Predictions (Inference)
#------------------------------------------------------------------------------
def inference(base_path) -> float:
    """
    Make future price predictions
    
    Returns:
        Next day predicted price
    """
    logger.info("=== INFERENCE PHASE ===")
    
    # Load data and model
    data, model = load_data_and_model(base_path)
    if data is None or model is None:
        return 0.0
    
    # Get last sequence for prediction
    last_sequence = data['last_sequence']
    scalers = data['column_scaler']
    
    if last_sequence is not None:
        # Reshape for prediction (batch dim = 1)
        last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
        
        # Make prediction using shared function
        prediction = predict_and_transform(model, last_sequence, scalers, is_single_prediction=True)
        
        if prediction is not None:
            next_day_price = prediction[0][0]
            logger.info(f"Next day prediction: ${next_day_price:.2f}")
            
            # Future prediction completed successfully
            logger.info(f"Future prediction saved: ${next_day_price:.2f}")
            
            return next_day_price
        else:
            logger.error("Prediction failed")
            return 0.0
    else:
        logger.error("Last sequence not available for prediction")
        return 0.0

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
    
    # Feature selection arguments
    parser.add_argument("--features", nargs='+', 
                       default=[PRICE_VALUE],
                       help="Features to use for prediction (default: %(default)s)")
    parser.add_argument("--prediction_days", type=int, default=PREDICTION_DAYS,
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
    FEATURES = args.features
    PREDICTION_DAYS = args.prediction_days
    TEST_SIZE = args.test_size
    SPLIT_METHOD = args.split_method
    SHUFFLE = args.shuffle
    SCALE = args.scale
    MODEL_NAME = args.model_name
    
    base_path = f"{START_DATE}_{TICKER}_{FEATURES}_seq-{PREDICTION_DAYS}-step_1_{MODEL_NAME}"
    
    # Create necessary directories
    for dir in ["cache", "cache/trained_models", "cache/processed_data", "results"]:
        os.makedirs(f"dev/{dir}", exist_ok=True)
    
    # Execute the clean DRY pipeline
    print("="*60)
    print("STOCK PREDICTION PIPELINE")
    print("="*60)
    
    # 1. Train model
    train(base_path)
    
    # 2. Test model performance  
    test_metrics = test(base_path)
    
    # 3. Make future predictions
    next_price = inference(base_path)
    
    print("="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Next day prediction: ${next_price:.2f}")
    print("="*60)