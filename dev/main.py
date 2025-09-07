# PyTorch Training Pipeline
# Clean, GPU-optimized stock prediction using PyTorch only

import numpy as np
import pandas as pd
import os
from loguru import logger
from data_preprocessing.data_processor import DataProcessor
from data_preprocessing.data_splitting import val_split_with_prices
from utils.file_handling import init_cache_dir, save_data
from config.data import *
from utils.plots import plot_predictions
from argparse import ArgumentParser
from utils.evaluating_utils import calculate_trading_metrics_returns

# Import PyTorch models and trainers directly
from model.pytorch_models import pytorch_model_factory
from train.pytorch_trainer import fit, predict, evaluate, save_model, load_model, get_device


def create_pytorch_model(input_size, model_name, layers, dropout_rate):
    """Create PyTorch model directly."""
    logger.info(f"Creating PyTorch {model_name} model...")
    model = pytorch_model_factory(
        input_size=input_size,
        model_name=model_name,
        layers=layers,
        dropout_rate=dropout_rate
    )
    return model


def sanity_check(label, arr):
    """Sanity check helper for debugging scaling issues."""
    arr = np.array(arr)
    return f"{label}: mean={arr.mean():.3f}, std={arr.std():.3f}, min={arr.min():.3f}, max={arr.max():.3f}"

def predict_and_evaluate(model, x_test, y_test, target_scaler, device=None, current_prices=None, threshold=0.005, cost=0.001):
    """Make predictions and evaluate model using PyTorch functions."""
    logger.info("Making predictions using PyTorch...")
    
    # Get predictions
    predictions = predict(model, x_test, device)
    
    # Evaluate model performance
    try:
        loss_val = evaluate(model, x_test, y_test, device)
        logger.info(f"Model evaluation completed, loss: {loss_val}")
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        loss_val = None
    
    # Sanity check raw predictions
    logger.info(f"Raw predictions - {sanity_check('pred_scaled', predictions)}")
    logger.info(f"Raw targets - {sanity_check('true_scaled', y_test)}")
    
    # Inverse transform using dedicated target scaler
    if target_scaler is not None:
        actual_prices = target_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).reshape(-1)
        predicted_prices = target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(-1)
        logger.info(f"Inverse transformed using dedicated target scaler")
        
        # Sanity check inverse transformed predictions
        logger.info(f"Inverse predictions - {sanity_check('pred_price', predicted_prices)}")
        logger.info(f"Inverse targets - {sanity_check('true_price', actual_prices)}")
        
        # Build current prices from the feature scaler (close is index 0)
        close_idx = 0
        if current_prices is None or len(current_prices) != len(y_test):
            curr_close_scaled = x_test[:, -1, close_idx].reshape(-1, 1)
            # Use the feature scaler for current prices (from input features)
            feature_scaler = None
            # We'll get this from the calling function
            current_prices = curr_close_scaled.reshape(-1)  # Will be fixed in caller
    else:
        # No scaling was applied, use values as-is
        actual_prices = y_test.reshape(-1)
        predicted_prices = predictions.reshape(-1)
        # Validate or rebuild current prices from x_test directly
        close_idx = 0
        if current_prices is None or len(current_prices) != len(y_test):
            current_prices = x_test[:, -1, close_idx].reshape(-1)
        else:
            current_prices = current_prices.reshape(-1)
        logger.info("No inverse transform applied - values used as-is")

    # Defensive check
    assert len(actual_prices) == len(predicted_prices) == len(current_prices), (
        f"Length mismatch: actual={len(actual_prices)}, pred={len(predicted_prices)}, curr={len(current_prices)}"
    )
    
    # Use clean price-based evaluation (Rule B, C)
    from utils.evaluating_utils import trading_metrics_from_prices
    
    # Create price sequences for evaluation 
    p_true_seq = np.concatenate([current_prices[:1], actual_prices])
    p_pred_seq = np.concatenate([current_prices[:1], predicted_prices])
    
    # Add diagnostics (Rule D)
    actual_returns = (actual_prices - current_prices) / current_prices
    predicted_returns = (predicted_prices - current_prices) / current_prices
    corr = np.corrcoef(predicted_returns, actual_returns)[0,1] if len(predicted_returns) > 1 else 0.0
    logger.info(f"Correlation(pred_ret, true_ret) = {corr:.3f}")
    
    # Clean evaluation with consistent data paths
    trading_metrics = trading_metrics_from_prices(
        p_true_seq, p_pred_seq, threshold=threshold, cost=cost
    )
    
    # Convert to old format for CSV compatibility 
    trading_metrics_compat = {
        'accuracy_score': trading_metrics['accuracy'],
        'total_profit': trading_metrics['total_profit_pct'] * 10000,  # $10k notional
        'profit_per_trade': trading_metrics['profit_per_trade_pct'],
        'profitable_trades': trading_metrics['profitable_trades'],
        'total_trades': trading_metrics['total_trades'],
        'total_samples': len(actual_prices)
    }
    
    logger.info(f"Trading Performance: Profit=${trading_metrics_compat['total_profit']:.2f}, "
               f"Accuracy={trading_metrics['accuracy']:.3f}, "
               f"Trades={trading_metrics['total_trades']}")
        
    return actual_prices, predicted_prices, loss_val, trading_metrics_compat

#------------------------------------------------------------------------------
# Train Model
#------------------------------------------------------------------------------
def train_model(base_path, device=None) -> dict:
    """PyTorch training function."""
    logger.info("=== TRAINING PHASE ===")
    
    # Check if data already processed
    data_path = f"cache/processed_data/{base_path}.pkl"
    if not os.path.exists(data_path):
        # Create DataProcessor instance
        processor = DataProcessor(cache_dir='cache')
        
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
            save_data(data, f"cache/processed_data/{base_path}.pkl")
        else: 
            logger.error("Data processing failed")
            exit(1)
    else:
        logger.info("Loading cached processed data")
        data = pd.read_pickle(data_path)

    # Extract data
    x_train = data['X_train']
    y_train = data['y_train']
    x_test = data['X_test']
    y_test = data['y_test']
    current_prices_train = data['current_prices_train']
    current_prices_test = data['current_prices_test']
    
    logger.info(f"Training data shape: x_train={x_train.shape}, y_train={y_train.shape}")
    logger.info(f"Test data shape: x_test={x_test.shape}, y_test={y_test.shape}")

    # Create model
    logger.info("Building PyTorch model...")
    model = create_pytorch_model(
        input_size=x_train.shape[2],  # Number of features
        model_name=MODEL_NAME.lower(),
        layers=LAYERS,
        dropout_rate=DROPOUT_RATE
    )
    
    # Determine model path (for caching)
    model_path = f"cache/trained_models/{base_path}.pth"

    # Attempt to load cached trained model first
    loaded_from_cache = False
    try:
        if os.path.exists(model_path):
            model = load_model(model.__class__, model_path, device)
            loaded_from_cache = True
            logger.info(f"Loaded cached PyTorch model from: {model_path}")
    except Exception as e:
        logger.warning(f"Could not load cached model, will retrain: {e}")

    # Train and save if not loaded from cache
    if not loaded_from_cache:
        logger.info("Training model...")
        
        # Create validation split (20% of training data)
        x_train, y_train, x_val, y_val, _, _ = val_split_with_prices(x_train, y_train, current_prices_train, val_size=0.2)
        
        model = fit(model, x_train, y_train, x_val, y_val,
                   epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, patience=20, device=device)
        save_model(model, model_path)
        logger.info(f"Model saved to: {model_path}")

    logger.info("=== TESTING PHASE ===")
    
    target_feature = data['target_feature'].lower()
    # Use dedicated target scaler instead of feature scaler
    target_key = f"{TICKER}_{target_feature}_target"
    target_scaler = data['scalers'].get(target_key)
    
    # Get feature scaler for current prices
    feature_key = f"{TICKER}_{target_feature}"
    feature_scaler = data['scalers'].get(feature_key)
    
    logger.info(f"Using target scaler: {target_scaler is not None}")
    logger.info(f"Using feature scaler: {feature_scaler is not None}")
    
    # Use best threshold/cost from finetune if available
    threshold = 0.005
    cost = 0.0005
    try:
        best_overall_path = 'cache/finetune/best_overall_pytorch_config.json'
        if os.path.exists(best_overall_path):
            with open(best_overall_path, 'r') as f:
                best_overall = json.load(f)
            hp = best_overall.get('best_hyperparameters', {})
            threshold = hp.get('threshold', threshold)
            cost = hp.get('cost', cost)
    except Exception as e:
        logger.warning(f"Could not load best threshold/cost from finetune: {e}")

    # Fix current prices using feature scaler if needed
    current_prices_test = data.get('current_prices_test')
    if current_prices_test is None and feature_scaler is not None:
        # Extract current prices from x_test using feature scaler
        close_idx = 0  # 'close' is index 0
        curr_close_scaled = x_test[:, -1, close_idx].reshape(-1, 1)
        current_prices_test = feature_scaler.inverse_transform(curr_close_scaled).reshape(-1)
        logger.info("Reconstructed current prices from x_test using feature scaler")
    
    # Make predictions
    actual_prices, predicted_prices, loss_val, trading_metrics = predict_and_evaluate(
        model, x_test, y_test, target_scaler, device,
        current_prices=current_prices_test,
        threshold=threshold, cost=cost
    )
    
    # Generate plots
    test_df = data.get('test_df', pd.DataFrame())
    test_dates = test_df.index
    plot_path = f"results/{base_path}_predictions_chart.png"
    plot_predictions(actual_prices, predicted_prices, TICKER, save_path=plot_path, dates=test_dates)

    # Export returns-based trading metrics to CSV (clean summary)
    os.makedirs('results', exist_ok=True)
    csv_filename = f"results/{base_path}.csv"
    try:
        summary_keys = [
            'accuracy_score', 'total_profit', 'profit_per_trade',
            'profitable_trades', 'total_trades', 'total_samples'
        ]
        report = {k: trading_metrics.get(k) for k in summary_keys}
        report.update({
            'ticker': TICKER,
            'model_name': MODEL_NAME,
            'dropout_rate': DROPOUT_RATE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'threshold': threshold,
            'cost': cost,
            'latest_predicted_price': float(predicted_prices[-1]) if len(predicted_prices) else None,  # Rule E: inverse-transformed
            'loss_val': float(loss_val) if loss_val is not None else None,
        })
        pd.DataFrame([report]).to_csv(csv_filename, index=False)
        logger.info(f"Test results saved to: {csv_filename}")
    except Exception as e:
        logger.error(f"Failed to save CSV report: {e}")

    # Also return the metrics for programmatic use
    return trading_metrics


#------------------------------------------------------------------------------
# Command Line Arguments
#------------------------------------------------------------------------------
def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description='Simple Stock Price Prediction')
    
    # Data arguments
    parser.add_argument("--company", type=str, default=COMPANY,
                        help="Company ticker symbol (default: %(default)s)")
    parser.add_argument("--start_date", type=str, default=TRAIN_START,
                        help="Start date for data (default: %(default)s)")
    parser.add_argument("--end_date", type=str, default=TRAIN_END,
                        help="End date for data (default: %(default)s)")
    
    # Feature selection arguments (using all OHLCV features, predicting Close price)
    parser.add_argument("--target_feature", type=str, default=PRICE_VALUE,
                        help="Target feature to predict (default: Close)", 
                        choices=["Close", "Open", "High", "Low", "AdjClose", "Volume"])
    parser.add_argument("--lag_days", type=int, default=LAG_DAYS,
                        help="Number of days to look back for prediction (default: %(default)s)")
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="Test size ratio (default: %(default)s)")
    parser.add_argument("--split_method", type=str, default="date",
                        choices=['date', 'random'],
                        help="Data splitting method (default: %(default)s)")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="Shuffle data during training")
    parser.add_argument("--scale", action="store_true", default=True,
                        help="Scale the data")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default=None, 
                        help="Model type to use", 
                        choices=['lstm','gru','rnn','bilstm']) # if none -> load the best fined-tune model
    parser.add_argument("--layers", type=str, default="64,32",
                        help="Layer sizes as comma-separated list (e.g., '64,32,16')")
    parser.add_argument("--dropout_rate", type=float, default=0.2,
                        help="Dropout rate for regularization (default: 0.2)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    
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
    MODEL_NAME = args.model_name or 'lstm'  # Default to LSTM
    TARGET_FEATURE = args.target_feature
    
    # Parse hyperparameters
    LAYERS = [int(x.strip()) for x in args.layers.split(',')]
    DROPOUT_RATE = args.dropout_rate
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    init_cache_dir()
    
    # Best config loading - only when user provides minimal parameters
    import json
    import sys
    
    # Check if user provided explicit parameters (not just defaults)
    user_provided_params = any([
        '--layers' in sys.argv,  # User specified layers
        '--dropout_rate' in sys.argv,  # User specified dropout
        '--learning_rate' in sys.argv,  # User specified learning rate
        '--epochs' in sys.argv,  # User specified epochs
        '--batch_size' in sys.argv  # User specified batch size
    ])
    
    best_config = None
    config_source = None
    
    # Only load best config if user provided minimal parameters
    if not user_provided_params:
        overall_config_path = "cache/finetune/best_overall_pytorch_config.json"
        
        if os.path.exists(overall_config_path):
            with open(overall_config_path, 'r') as f:
                config_data = json.load(f)
            
            if MODEL_NAME is None:
                # Load overall best config
                best_config = config_data['best_hyperparameters']
                config_source = f"overall best config ({config_data['best_model_name'].upper()})"
                score = config_data.get('best_score', 'N/A')
                logger.info(f"Using {config_source} (Score: {score})")
            else:
                # Load model-specific best config
                if MODEL_NAME in config_data.get('all_results', {}):
                    model_results = config_data['all_results'][MODEL_NAME]
                    best_config = model_results['best_hyperparameters']
                    config_source = f"best {MODEL_NAME.upper()} config"
                    score = model_results.get('best_score', 'N/A')
                    logger.info(f"Using {config_source} (Score: {score})")
                else:
                    logger.info(f"No best config found for {MODEL_NAME.upper()}, using default parameters")
        else:
            logger.info("No best config files found, using default parameters")
        
        # Apply best config if found
        if best_config:
            n_layers = best_config.get('n_layers', 2)
            LAYERS = [best_config[f'units_l{i}'] for i in range(n_layers)]
            if MODEL_NAME is None:
                MODEL_NAME = config_data.get('best_model_name', MODEL_NAME)
            DROPOUT_RATE = best_config.get('dropout_rate', DROPOUT_RATE)
            LEARNING_RATE = best_config.get('learning_rate', LEARNING_RATE)
            EPOCHS = best_config.get('epochs', EPOCHS)
            BATCH_SIZE = best_config.get('batch_size', BATCH_SIZE)
    else:
        logger.info("User provided explicit parameters, using command-line arguments (best config ignored)")

    logger.info(f"Using parameters: \n\tModel: {MODEL_NAME.upper() if MODEL_NAME else 'None'}, \n\tLayers: {LAYERS}, \n\tDropout: {DROPOUT_RATE}, \n\tLearning Rate: {LEARNING_RATE}, \n\tOptimizer: AdamW, \n\tEpochs: {EPOCHS}, \n\tBatch Size: {BATCH_SIZE}")

    # Detect device once at the start
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create base path
    layers_str = '-'.join(map(str, LAYERS))
    lr_str = f"{LEARNING_RATE:.4f}".rstrip('0').rstrip('.')
    dropout_str = f"{DROPOUT_RATE:.1f}".rstrip('0').rstrip('.')
    base_path = f"{TICKER}_{START_DATE}_to_{END_DATE}_{TARGET_FEATURE}_seq-{LAG_DAYS}-step_1_{MODEL_NAME}_layers{layers_str}_dropout{dropout_str}_lr{lr_str}_adamw_epochs{EPOCHS}_bs{BATCH_SIZE}"
    
    # Create directories
    for dir in ["cache", "cache/trained_models", "cache/processed_data", "cache/raw_data", "cache/scalers", "results"]:
        os.makedirs(f"{dir}", exist_ok=True)
    
    # Train model
    print("="*60)
    print("SIMPLE STOCK PREDICTION PIPELINE")
    print("="*60)
    
    train_model(base_path, device)
    
    print("="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)