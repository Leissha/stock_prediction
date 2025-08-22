# File: stock_prediction.py (Enhanced Version)
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

import numpy as np
from loguru import logger
from model import *
from data_preprocessing.data_processor import DataProcessor
from utils.file_handling import ensure_directory_exists
from config.data import *
from utils.plots import plot_predictions
from argparse import ArgumentParser
from model.lstm import LSTMModel
from model.bidirectional_lstm import BidirectionalLSTMModel

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


#------------------------------------------------------------------------------
# Train model
#------------------------------------------------------------------------------
def train():
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
    
    # Create necessary directories
    for dir in ["trained_models", "cache", "results"]:
        ensure_directory_exists(f"dev/{dir}")
    
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
        scale=SCALE
    )
    
    # Extract the processed data
    x_train = data['X_train']
    y_train = data['y_train']
    x_test = data['X_test']
    y_test = data['y_test']
    scalers = data['column_scaler']
    
    logger.info(f"Training data shape: x_train={x_train.shape}, y_train={y_train.shape}")
    logger.info(f"Test data shape: x_test={x_test.shape}, y_test={y_test.shape}")
    
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
    target_feature = args.features[0]  # Use first feature as target
    model_filename = f"dev/trained_models/{args.company}_{target_feature}_{args.prediction_days}_{MODEL_NAME}.h5"
    model.save_model(model_filename)
    logger.info(f"Model saved to: {model_filename}")
    
    # 3. Make predictions on test data
    logger.info("=== Making Predictions ===")
    predicted_prices = model.predict(x_test)
    
    # Model predicts in scaled space, so we need to inverse transform predictions
    target_feature = args.features[0]  # Use first feature as target
    if predicted_prices is not None and target_feature in scalers:
        # Inverse transform predictions from scaled to original scale
        predicted_prices = scalers[target_feature].inverse_transform(predicted_prices.reshape(-1, 1)).flatten()
        actual_prices = y_test  # y_test is already in original scale
    else:
        actual_prices = y_test
        if predicted_prices is None:
            logger.error("Model prediction failed")
            return
    
    # 4. Plot results
    plot_predictions(actual_prices, predicted_prices, 
                    f"dev/results/{args.company}_{target_feature}_{args.prediction_days}_predictions.png")
    
    # 5. Predict next day using last sequence
    logger.info("=== Next Day Prediction ===")
    last_sequence = data['last_sequence']
    if last_sequence is not None:
        # Reshape for prediction (add batch dimension)
        last_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
        
        prediction = model.predict(last_sequence)
        if prediction is not None and target_feature in scalers:
            # Inverse transform prediction from scaled to original scale
            prediction = scalers[target_feature].inverse_transform(prediction.reshape(-1, 1))
            logger.info(f"Next day prediction: {prediction[0][0]:.2f}")
        else:
            logger.error("Next day prediction failed")
    else:
        logger.error("Last sequence not available for prediction")
    
    # 6. Print some evaluation metrics
    logger.info("=== Model Evaluation ===")
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    
    logger.info(f"Mean Absolute Error: {mae:.2f}")
    logger.info(f"Root Mean Square Error: {rmse:.2f}")
    
    sc = scalers[target_feature]
    logger.info(f"Scaler min/max for {target_feature}: {sc.data_min_[0]}, {sc.data_max_[0]}")
    logger.info(f"predicted (scaled) range: {predicted_prices.min():.4f}-{predicted_prices.max():.4f}")
    logger.info(f"y_test (scaled) range: {y_test.min():.4f}-{y_test.max():.4f}")
    
    # 7. Calculate trading-based accuracy metrics (same as v0.1 and p1)
    logger.info("=== Trading Accuracy Metrics ===")
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.accuracy_utils import calculate_trading_accuracy, print_accuracy_metrics, save_accuracy_to_csv
        
        # Get current prices for trading accuracy calculation
        # We need to get the current prices (day before each prediction)
        # This depends on how the data was processed in DataProcessor
        current_prices = data.get('current_prices', actual_prices)  # Fallback to actual_prices if not available
        
        # Calculate trading accuracy
        metrics = calculate_trading_accuracy(
            actual_prices=actual_prices,
            predicted_prices=predicted_prices,
            current_prices=current_prices,
            lookup_step=1  # dev predicts next day
        )
        
        # Get next day prediction for CSV output
        next_day_predicted = prediction[0][0] if prediction is not None else 0
        
        # Print metrics
        print_accuracy_metrics(
            metrics=metrics,
            future_price=next_day_predicted,
            loss_value=mse,
            loss_name="mean_squared_error loss",
            lookup_step=1
        )
        
        # Save to CSV
        csv_filename = f"dev/results/{args.company}_{target_feature}_{args.prediction_days}_accuracy.csv"
        save_accuracy_to_csv(
            metrics=metrics,
            future_price=next_day_predicted,
            loss_value=mse,
            loss_name="mean_squared_error loss",
            filename=csv_filename,
            lookup_step=1
        )
        logger.info(f"Accuracy metrics saved to: {csv_filename}")
        
    except ImportError as e:
        logger.warning(f"Could not import accuracy_utils: {e}")
        logger.info("Using standard metrics only")
    except Exception as e:
        logger.error(f"Error calculating trading accuracy: {e}")
        logger.info("Using standard metrics only")

if __name__ == "__main__":
    train()
