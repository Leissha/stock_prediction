#!/usr/bin/env python3
"""
Inference script to load trained model and make predictions
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os
from loguru import logger
import pickle

def load_trained_model(model_name="cba_stock_prediction"):
    """Load the trained model and related data"""
    
    # Load the full model
    model_path = os.path.join("results", f"{model_name}_full_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = load_model(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Load results (if available)
    results_path = os.path.join("results", f"{model_name}_results.npy")
    if os.path.exists(results_path):
        results = np.load(results_path, allow_pickle=True).item()
        logger.info(f"Loaded results from {results_path}")
    else:
        logger.warning(f"Results file not found at {results_path}")
        results = {}
    
    # Load training history
    history_path = os.path.join("results", f"{model_name}_training_history.npy")
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        logger.info(f"Loaded training history from {history_path}")
    else:
        history = None
    
    return model, results, history

def load_scaler(model_name="cba_stock_prediction"):
    """Load the scaler used during training"""
    scaler_path = os.path.join("trained_models", f"{model_name}_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {scaler_path}")
        return scaler
    else:
        logger.warning(f"Scaler not found at {scaler_path}")
        return None

def predict_next_day(model, last_sequence, scaler=None):
    """Make next day prediction"""
    # Reshape for prediction
    prediction_input = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
    
    # Make prediction
    prediction = model.predict(prediction_input, verbose=0)
    
    # Inverse transform if scaler is available
    if scaler is not None:
        prediction = scaler.inverse_transform(prediction)
    
    return prediction[0][0]

def evaluate_model(model, X_test, y_test, scaler=None):
    """Evaluate model performance"""
    # Make predictions
    predictions = model.predict(X_test, verbose=0)
    
    # Inverse transform if scaler is available
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    else:
        actual_prices = y_test.reshape(-1, 1)
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(actual_prices, predictions)
    mse = mean_squared_error(actual_prices, predictions)
    rmse = np.sqrt(mse)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'predictions': predictions.flatten(),
        'actual_prices': actual_prices.flatten()
    }

def main():
    """Main inference function"""
    model_name = "cba_stock_prediction"
    
    try:
        # Load model and data
        logger.info("=== Loading Trained Model ===")
        model, results, history = load_trained_model(model_name)
        
        # Load scaler
        scaler = load_scaler(model_name)
        
        # Print model summary
        logger.info("Model Summary:")
        model.summary()
        
        # Print training results if available
        if results:
            logger.info("Training Results:")
            logger.info(f"  MAE: {results['mae']:.2f}")
            logger.info(f"  RMSE: {results['rmse']:.2f}")
            
            # Print training parameters
            if 'training_params' in results:
                params = results['training_params']
                logger.info("Training Parameters:")
                for key, value in params.items():
                    logger.info(f"  {key}: {value}")
        
        # Example: Load test data for evaluation
        data_path = os.path.join("data", "CBA.AX_data.csv")
        if os.path.exists(data_path):
            logger.info("Loading test data for evaluation...")
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # You can add more evaluation code here
            # For now, just show the data
            logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
            logger.info(f"Price range: {df['adjclose'].min():.2f} to {df['adjclose'].max():.2f}")
        
        logger.info("=== Inference Ready ===")
        logger.info("Model loaded successfully and ready for predictions!")
        
        # Example of how to use the model for predictions
        logger.info("To make predictions, use:")
        logger.info("  prediction = predict_next_day(model, last_sequence, scaler)")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Make sure to run train.py first to train the model")

if __name__ == "__main__":
    main()
