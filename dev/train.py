# File: train.py
# Authors: Kha Anh Nguyen :)
# Date: 24/08/2025

import numpy as np
import pandas as pd
import os
from loguru import logger
from data_preprocessing.data_processor import DataProcessor
from utils.file_handling import save_data
from config.data import *
from utils.plots import plot_predictions
from argparse import ArgumentParser
from model.tf_models import TFModel
from utils.evaluating_utils import calculate_trading_metrics
import tensorflow as tf    

def predict_and_descale(model, x_test, y_test, target_scaler=None):
    """
    Simplified prediction function with single-scaler inverse transform
    
    Args:
        model: Trained model
        x_test: Test input data
        y_test: Test target data (scaled)
        data: Dictionary containing scalers and metadata from DataProcessor

    Returns:
        tuple: (actual_prices, predicted_prices, metrics) - all in original scale
    """
    # Make predictions using model
    predictions, metrics = model.predict_and_evaluate(x_test, y_test)
    
    # Inverse transform using separate scaler for target feature
    if target_scaler is not None:
        actual_prices = target_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).reshape(-1)
        predicted_prices = target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(-1)
        
        print(f"Inverse transformed using scaler: {target_scaler}")
    else:
        # No scaling was applied, use values as-is
        actual_prices = y_test.reshape(-1)
        predicted_prices = predictions.reshape(-1)
        print("No inverse transform applied - values used as-is")
        
    return actual_prices, predicted_prices, metrics

#------------------------------------------------------------------------------
# Train Model
#------------------------------------------------------------------------------
def train(args) -> dict:
    # data parameters:
    ticker = args.company
    start_date = args.start_date
    end_date = args.end_date
    lag_days = args.lag_days
    test_size = args.test_size
    split_method = args.split_method
    shuffle = args.shuffle
    scale = args.scale

    # model parameters:
    model_name = (args.model_name or 'lstm')
    layers = args.layers
    dropout_rate = args.dropout_rate
    epochs = args.epochs
    batch_size = args.batch_size

    # prediction target:
    target_feature = args.target_feature
    # --log_ret also calls --target_ret
    use_log_returns = args.log_ret
    target_as_return = args.target_ret or args.log_ret

    # Paths:
    ret_suffix = "_ret" if target_as_return else ""
    base_path = f"{ticker}_{start_date}_to_{end_date}_{target_feature}{ret_suffix}"
    data_path = f"cache/processed_data/{base_path}.pkl"
    
    meta_path = f"{base_path}_seq-{lag_days}-step_1_{model_name}_layers{layers}_dropout{dropout_rate}_epochs{epochs}_bs{batch_size}"
    model_path = f"cache/trained_models/{meta_path}.keras"
    
    plot_path = f"results/{meta_path}_predictions_chart.png"
    report_path = f"results/{meta_path}.csv"
    
    # Create necessary directories
    for dir in ["cache", "cache/trained_models", "cache/processed_data", "cache/raw_data", "cache/scalers", "results"]:
        os.makedirs(f"{dir}", exist_ok=True)

    print("="*60)
    print("STOCK PREDICTION PIPELINE")
    print("="*60)

    print("=== TRAINING PHASE ===")
    
    # Check if data already processed
    if not os.path.exists(data_path):
        # Create DataProcessor instance
        processor = DataProcessor(cache_dir='cache')
        
        # Use the comprehensive data processing method
        data = processor.data_processing(
            start_date=start_date,
            end_date=end_date,
            ticker=ticker,
            lag_days=lag_days,
            lookup_step=1,
            shuffle=shuffle,
            splitting_method=split_method,
            test_size=test_size,
            target_feature=target_feature,
            scale=scale,
            target_as_return=target_as_return,
            use_log_returns=use_log_returns,
        )
        
        if data:
            save_data(data, f"{data_path}")
        else: 
            logger.error("Data processing failed")
            exit(1)
    else:
        print("Loading cached processed data")
        data = pd.read_pickle(data_path)

    # Extract the processed data
    x_train = data['X_train']
    y_train = data['y_train']
    x_test = data['X_test']
    y_test = data['y_test']
    
    print(f"Training data shape: x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"Test data shape: x_test={x_test.shape}, y_test={y_test.shape}")

    # Init model instance
    input_size = x_train.shape[2] # no_features
    tf_model = TFModel(
        input_size=input_size,
        model_name=model_name,
        layers=layers,
        dropout_rate=dropout_rate,
    )
    
    # Check if model already trained
    if not os.path.exists(model_path):
        print("Building and training model...")
        # Train the model
        tf_model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Save the trained model
        tf_model.save_model(f"{model_path}")
        print(f"Model saved to: {model_path}")

        # Use the underlying Keras model
        model = tf_model
    else:
        print("Using existing trained model")
        # Load the existing Keras model and wrap it in TFModel
        keras_model = tf.keras.models.load_model(f"{model_path}")  # type: ignore
        
        tf_model.model = keras_model  # Replace the built model with loaded one
        model = tf_model

    print("=== TESTING PHASE ===")
    
    tf_key = data['target_feature'].lower()
    target_key = f"{ticker}_{tf_key}"
    target_scaler = data['scalers'].get(target_key)
    
    # Make predictions on test set
    actual_prices, predicted_prices, metrics = predict_and_descale(model, x_test, y_test, target_scaler)

    # If model predicts returns, reconstruct next-day prices point-wise (no compounding)
    if target_as_return:
        test_df = data.get('test_df', pd.DataFrame()).copy()
        
        # Get base price column (target: 'close_return' -> get training 'close' price for first prediction)
        base_price_col = tf_key.replace('_return', '')
        
        if base_price_col in test_df.columns:
            # true future prices aligned to y_test
            true_prices = test_df[base_price_col].values[-len(actual_prices):]
            
            # Use last available training-day price to derive first test prediction to avoid leakage
            last_training_price = float(data.get('last_training_price', test_df[base_price_col].iloc[0]))
            
            # Use true previous prices as base
            preds_ret = predicted_prices.reshape(-1)
            
            # Current prices for each step: day 0 uses seed, other days use TRUE previous prices
            current_prices = [last_training_price] + list(true_prices[:-1])
            
            if use_log_returns:
                # Log returns: p_t = p_{t-1} * exp(r_t)
                predicted_prices = np.array(current_prices) * np.exp(preds_ret)
            else:
                # Simple returns: p_t = p_{t-1} * (1 + r_t)
                predicted_prices = np.array(current_prices) * (1.0 + preds_ret)
            actual_prices = true_prices
        else:
            logger.warning(f"Base price column '{tf_key}' not in test_df; available sample columns: {list(test_df.columns)[:8]}")
    
    # Generate plots
    test_df = data.get('test_df', pd.DataFrame())
    test_dates = test_df.index
    plot_predictions(actual_prices, predicted_prices, ticker, save_path=plot_path, dates=test_dates)

    # Calculate metrics and save to CSV
    # Get the latest prediction as a plain float for reporting
    future_price = float(predicted_prices[-1])
    
    # Get current prices (previous day's actual prices for realistic trading)
    current_prices = []
    for i in range(len(actual_prices)):
        if i == 0:
            current_prices.append(actual_prices[i])
        else:
            current_prices.append(actual_prices[i-1])
    
    # Get training features from data
    training_features = data['training_features']
    
    metrics = calculate_trading_metrics(
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        current_prices=current_prices,
        lookup_step=1,
        future_price=future_price,
        loss_val=metrics['loss'],
        mae_val=metrics['mae'],
        rmse_val=metrics['rmse'],
        filename=report_path,
        scale=scale,
        target_feature=data['target_feature'],
        training_features=training_features,
    )
    
    print(f"Test results saved to: {report_path}")
    print("="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    return metrics

#------------------------------------------------------------------------------
# Command Line Arguments
#------------------------------------------------------------------------------
def parse_args():
    """Parse command line arguments for the stock prediction model"""
    parser = ArgumentParser(description='Stock Price Prediction with LSTM')
    
    # Data loading arguments
    parser.add_argument("--company", type=str, default=TICKER,
                       help="Company ticker symbol (default: %(default)s)")
    parser.add_argument("--start_date", type=str, default=START_DATE,
                       help="Start date for data (default: %(default)s)")
    parser.add_argument("--end_date", type=str, default=END_DATE,
                       help="End date for data (default: %(default)s)")
    
    # Feature selection arguments (using all OHLCV features, predicting Close price)
    parser.add_argument("--target_feature", type=str, default=TARGET_FEATURE,
                       help="Target feature to predict (default: Close)", choices=["Close", "Open", "High", "Low", "AdjClose", "Volume"])
    parser.add_argument("--target_ret", action="store_true", default=False,
                       help="Predict simple percentage returns instead of raw prices")
    parser.add_argument("--log_ret", action="store_true", default=False,
                       help="Predict log returns instead of raw prices")
    parser.add_argument("--lag_days", type=int, default=LAG_DAYS,
                       help="Number of days to look back for prediction (default: %(default)s)")
    
    # Data processing arguments
    parser.add_argument("--test_size", type=float, default=TEST_SIZE,
                       help="Test size ratio (default: %(default)s)")
    parser.add_argument("--split_method", type=str, default=SPLIT_METHOD,
                       choices=['date', 'random'],
                       help="Data splitting method (default: %(default)s)")
    parser.add_argument("--shuffle", action="store_true", default=SHUFFLE,
                       help="Shuffle data during training")
    parser.add_argument("--scale", action="store_true", default=SCALE,
                       help="Scale the data")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default='lstm', 
                        help="Model type to use", 
                        choices=['lstm','gru','rnn','bilstm']) 
    parser.add_argument("--layers", nargs='+', type=int, default=LAYERS,
                        help="Layer sizes separated by space (e.g: 64 32 16 128)")
    parser.add_argument("--dropout_rate", type=float, default=DROPOUT,
                        help="Dropout rate for regularization (default: 0.2)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for training (default: 32)")
    
    return parser.parse_args()

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create necessary directories
    for dir in ["cache", "cache/trained_models", "cache/processed_data", "cache/raw_data", "cache/scalers", "results"]:
        os.makedirs(f"{dir}", exist_ok=True)
    
    # Execute the clean DRY pipeline
    print("="*60)
    print("STOCK PREDICTION PIPELINE")
    print("="*60)
    
    train(args)