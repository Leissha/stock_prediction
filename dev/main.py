"""
Financial AI Pipeline - Clean Main Entry Point

This module provides a unified entry point for all 7 execution modes:
1. Sentiment Analysis
2. TF Models (LSTM, GRU, RNN, BiLSTM)  
3. SARIMAX
4. Classification
5. Ensemble
6. CNN-LSTM Hybrid
7. Attention-LSTM
"""

import os
import sys

# Add the `dev` directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from argparse import ArgumentParser
from config.data import *
from config.pipeline_config import update_config_from_args, validate_current_config, print_current_config
from pipeline import prepare_data
from eval.regression_evaluator import run_regression_evaluation
from eval.classification_evaluator import run_classification_evaluation
import tensorflow as tf

def ensure_directories():
    """Ensure all required directories exist."""
    for dir in ["results", "cache/trained_models", "cache/sentiment", "cache/raw_data", "cache/processed_data", "cache/scalers"]:
        os.makedirs(dir, exist_ok=True)

# Define model sets before parse_args() so they're available for argument choices
ATTENTION_MODELS = {"attention_lstm", "attention_gru", "attention_rnn", "attention_bilstm"}
CNN_MODELS = {"cnn_lstm", "cnn_rnn", "cnn_gru", "cnn_bilstm"}
TF_BASE_MODELS = {"lstm", "gru", "rnn", "bilstm"}
TF_MODELS = ATTENTION_MODELS | CNN_MODELS | TF_BASE_MODELS

def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(description='Financial AI Pipeline - Stock Prediction & Analysis')

    # Core data arguments
    parser.add_argument("--company", type=str, default=TICKER,
                        help="Company ticker symbol (default: %(default)s)")
    parser.add_argument("--start_date", type=str, default=START_DATE,
                        help="Start date for data (default: %(default)s)")
    parser.add_argument("--end_date", type=str, default=END_DATE,
                        help="End date for data (default: %(default)s)")
    parser.add_argument("--target_feature", type=str, default=TARGET_FEATURE,
                        help="Target feature to predict (default: Close)",
                        choices=["Close", "Open", "High", "Low", "AdjClose", "Volume"])
    
    # Data processing arguments
    parser.add_argument("--lookback", type=int, default=LOOKBACK,
                        help="Number of days to look back (default: %(default)s)")
    parser.add_argument("--horizon", type=int, default=HORIZON,
                        help="Number of future days to predict (default: %(default)s)")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE,
                        help="Test size ratio (default: %(default)s)")
    parser.add_argument("--val_size", type=float, default=VAL_SIZE,
                        help="Validation size ratio (default: %(default)s)")
    parser.add_argument("--scale", action="store_true", default=SCALE,
                        help="Scale the data")
    
    # Target transformation arguments
    parser.add_argument("--target_ret", action="store_true", default=False,
                        help="Predict simple percentage returns instead of raw prices")
    parser.add_argument("--log_ret", action="store_true", default=False,
                        help="Predict log returns instead of raw prices")
    
    # Sentiment arguments
    parser.add_argument("--use_sentiment", action="store_true", default=False,
                        help="Include daily sentiment features if available")
    parser.add_argument("--include_social", action="store_true", default=False,
                        help="Include social media data (Reddit, StockTwits, Google Trends) in sentiment analysis")

    # Execution mode arguments
    parser.add_argument("--classification", action="store_true", default=False,
                        help="Run binary classification (up/down) evaluation and exit")
    parser.add_argument("--use_price_comparison", action="store_true", default=False,
                        help="For classification: compare predicted price with previous price instead of using threshold")
    parser.add_argument("--model_name", type=str, default='lstm',
                        help="Model type to use", choices=list(TF_MODELS) + ['sarimax', 'ensemble'])
    
    # TF Model arguments
    parser.add_argument("--layers", nargs='+', type=int, default=LAYERS,
                        help="Layer sizes separated by space (e.g: 64 32 16)")
    parser.add_argument("--dropout_rate", type=float, default=DROPOUT,
                        help="Dropout rate (default: 0.2)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size (default: 32)")
    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam','rmsprop','sgd'],
                        help="Optimizer for TF models (default: adam)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimizer (default: 1e-3)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (ReduceLROnPlateau) (default: 10)")
    # Attention-specific
    parser.add_argument("--attn_heads", type=int, default=4,
                        help="Number of attention heads for attention_* models (default: 4)")
    parser.add_argument("--attn_key_dim", type=int, default=16,
                        help="Key dimension per head for attention_* models (default: 16)")

    # CNN-specific arguments (for CNN hybrid models)
    parser.add_argument("--cnn_filters", nargs='+', type=int, default=[64, 32],
                        help="CNN filter counts separated by space (e.g: 64 32) (default: 64 32)")
    parser.add_argument("--cnn_kernel_size", type=int, default=3,
                        help="CNN kernel size (default: 3)")

    # SARIMAX arguments
    parser.add_argument("--sarimax_seasonal", action="store_true", default=False,
                        help="Enable seasonal SARIMAX")
    parser.add_argument("--sarimax_m", type=int, default=5,
                        help="Seasonal period m (default: 5)")
    
    # Ensemble arguments
    parser.add_argument("--ensemble_2", type=str, default='lstm',
                        help="Base TF model for ensemble (when --model_name ensemble)",
                        choices=list(TF_MODELS))
    parser.add_argument("--sarima_weight", type=float, default=0.5,
                        help="Weight for SARIMA in ensemble (default: 0.5)")
    parser.add_argument("--model_2_weight", type=float, default=0.5,
                        help="Weight for Model 2 in ensemble (default: 0.5)")
    
    # Utility arguments
    parser.add_argument("--inspect_plots", action="store_true", default=False,
                        help="Generate preprocessing inspection plots")

    return parser.parse_args()

def generate_meta_path(config):
    """Generate meta path for model saving and results"""
    # Determine target type suffix
    if config.data.use_log_returns:
        ret_suffix = "_log_ret"
    elif config.data.target_as_return:
        ret_suffix = "_ret"
    else:
        ret_suffix = ""

    # Determine model tag
    if config.mode == 'ensemble':
        meta_model_tag = f"ensemble+sarima-{config.ensemble.sarima_weight}+{config.ensemble.ensemble_2}-{config.ensemble.model_2_weight}"
    elif config.data.use_sentiment:
        meta_model_tag = f"sentiment+{config.tf_model.model_name}"
    else:
        meta_model_tag = f"{config.tf_model.model_name}"

    # Format layers as filename-safe string (replace brackets and commas with underscores)
    layers_str = "_".join(map(str, config.tf_model.layers))

    base_path = f"{config.data.company}_{config.data.start_date}_to_{config.data.end_date}_{config.data.target_feature}{ret_suffix}_step_{config.data.horizon}"
    meta_path = f"{base_path}_seq-{config.data.lookback}_{meta_model_tag}_layers{layers_str}_dropout{config.tf_model.dropout_rate}_epochs{config.tf_model.epochs}_bs{config.tf_model.batch_size}_patience{config.tf_model.patience}_lr{config.tf_model.learning_rate}"
    
    return meta_path

def create_model(config, bundle, meta_path, args):
    """Create and train/load model based on configuration"""
    input_size = bundle.X_train.shape[2]
    # Normalize mode: sentiment is a feature flag, not an execution mode
    if getattr(config, 'mode', None) == 'sentiment':
        config.mode = 'tf_model'

    if config.mode == "tf_model":
        return _create_tf_model(config, bundle, meta_path, input_size, args)
    elif config.mode == "sarimax":
        return _create_sarimax_model(config, bundle)
    elif config.mode == "ensemble":
        return _create_ensemble_model(config, bundle, meta_path, args)
    else:
        # Fallback to tf_model to avoid hard failures on unexpected modes
        return _create_tf_model(config, bundle, meta_path, input_size, args)

def _create_tf_model(config, bundle, meta_path, input_size, args):
    """Create and train/load TensorFlow model"""
    from model.tf_models import TFModel

    # Determine task type based on classification flag
    task_type = 'binary' if config.classification.enabled else 'regression'

    # Build any TFModel variant in one path
    model = TFModel(
        input_size=input_size,
        model_name=config.tf_model.model_name,
        layers=config.tf_model.layers,
        dropout_rate=config.tf_model.dropout_rate,
        output_steps=config.data.horizon,
        task_type=task_type,
        attn_heads=args.attn_heads,
        attn_key_dim=args.attn_key_dim
    )
    # Simple path construction (meta_path already sanitized)
    model_path = f"cache/trained_models/{config.data.company}_{config.tf_model.model_name}_{meta_path}.keras"

    if not os.path.exists(model_path):
        print(f"Training {config.tf_model.model_name.upper()} model...")
        val_tuple = None
        if bundle.X_val is not None and bundle.y_val is not None:
            val_tuple = (bundle.X_val, bundle.y_val)

        # Build training kwargs centrally
        fit_kwargs = {
            'epochs': config.tf_model.epochs,
            'batch_size': config.tf_model.batch_size,
            'validation_data': val_tuple,
            'meta_path': meta_path,
            'learning_rate': config.tf_model.learning_rate,
            'optimizer': config.tf_model.optimizer,
            'patience': config.tf_model.patience,
        }
        # Class weights balanced for classification
        if config.classification.enabled:
            from sklearn.utils.class_weight import compute_class_weight
            import numpy as np
            classes = np.unique(bundle.y_train.ravel())
            class_weights = compute_class_weight('balanced', classes=classes, y=bundle.y_train.ravel())
            fit_kwargs['class_weight'] = {int(c): float(w) for c, w in zip(classes, class_weights)}
            print(f"Class weights: {fit_kwargs['class_weight']}")

        model.fit(bundle.X_train, bundle.y_train, **fit_kwargs)

        # Save model (handle different interfaces robustly)
        try:
            model.save_model(model_path) 
        except Exception:
            try:
                model.save(model_path)  # type: ignore
            except Exception:
                model.model.save(model_path)  # type: ignore
        
        # Save scalers for inference reuse 
        # Scalers key name will be created from ticker, dates, target, scale, sentiment config. Features info already in the bundle file dictionary
        if bundle.scalers:
            from utils.file_handling import save_scalers, create_scaler_cache_key
            from schemas.bundle import TargetMode
            
            target_mode_str = (TargetMode.LOG_RETURN.value if config.data.use_log_returns 
                             else TargetMode.RETURN.value if config.data.target_as_return 
                             else TargetMode.PRICE.value)
            
            scaler_cache_key = create_scaler_cache_key(
                config.data.company, config.data.start_date, config.data.end_date,
                config.data.target_feature.lower(), target_mode_str,
                config.data.scale, config.data.use_sentiment
            )
            
            save_scalers(bundle.scalers, scaler_cache_key, cache_dir='cache')
            print(f"Scalers cached (shared across all models/configs with same data)")
    else:
        print(f"Loading existing model from {model_path}")
        from tensorflow import keras as tf_keras # type: ignore
        keras_model = tf_keras.models.load_model(model_path)
        model = keras_model
    
    return model

def _create_sarimax_model(config, bundle):
    """Create and train SARIMAX model"""
    from model.sarimax import SARIMAXModel
    
    print("Training SARIMAX model")
    model = SARIMAXModel(
        seasonal=config.sarimax.seasonal,
        m=config.sarimax.m,
    )

    # Use bundle (single source of truth)
    assert bundle.train_df is not None, "bundle.train_df missing"
    assert bundle.target_feature is not None, "bundle.target_feature missing"

    # Build data dict for SARIMA
    data_dict = {
        'train_df': bundle.train_df,
        'target_feature': bundle.target_feature,
        'scalers': bundle.scalers,
        'test_df': bundle.test_df,  # Optional for rolling eval
    }

    model.fit(data=data_dict)
    return model

def _create_ensemble_model(config, bundle, meta_path, args):
    """Create and train ensemble model"""
    from model.ensemble import EnsembleModel, EnsembleConfig
    
    # Ensemble weights configuration
    ensemble_config = EnsembleConfig(
        sarima_weight=config.ensemble.sarima_weight, 
        model_2_weight=config.ensemble.model_2_weight
    )
    # Create ensemble model
    model = EnsembleModel(ensemble_config)
    
    # TF model cache path (sanitize layers for filename)
    layers_str = "_".join(map(str, config.tf_model.layers))
    model_2_meta_path = f"{config.data.company}_{config.data.start_date}_to_{config.data.end_date}_{config.data.target_feature}_step_{config.data.horizon}_seq-{config.data.lookback}_{config.ensemble.ensemble_2}_layers{layers_str}_dropout{config.tf_model.dropout_rate}_epochs{config.tf_model.epochs}_bs{config.tf_model.batch_size}"
    
    # Extract only related args
    sarima_args = { 
        'seasonal': config.sarimax.seasonal, 
        'm': config.sarimax.m 
    }
    model_2 = { 
        'layers': config.tf_model.layers, 
        'dropout_rate': config.tf_model.dropout_rate, 
        'epochs': config.tf_model.epochs, 
        'batch_size': config.tf_model.batch_size, 
        'model_name': config.ensemble.ensemble_2, 
        'optimizer': config.tf_model.optimizer, 
        'learning_rate': config.tf_model.learning_rate,
        'attn_heads': args.attn_heads,
        'attn_key_dim': args.attn_key_dim
    }
    
    # Fit both models
    model.fit_from_bundle(bundle, sarima_args, model_2, model_2_meta_path)
    return model

def generate_inspection_plots(config, bundle, meta_path):
    """Generate optional inspection plots"""
    # Check if inspection plots are enabled
    if not hasattr(config, 'inspect_plots') or not config.inspect_plots:
        return
    
    try:
        # Create inspection data directory
        from utils.plots import create_candlestick_chart, create_boxplot
        os.makedirs("cache/inspect_data", exist_ok=True)
        # Check if train data is available
        if bundle.train_df is not None:
            create_candlestick_chart(
                bundle.train_df, 
                config.data.company, 
                save_path=f"cache/inspect_data/{meta_path}_candlestick.png", 
                n_days=1
            )
            create_boxplot(
                bundle.train_df, 
                config.data.company, 
                save_path=f"cache/inspect_data/{meta_path}_boxplot.png", 
                n_days=20
            )
    except Exception as e:
        print(f"Warning: Could not generate inspection plots: {e}")

def main():
    """Main execution function"""
    # Parse arguments and setup configuration
    args = parse_args()
    config = update_config_from_args(args)
    
    # Validate configuration
    errors = validate_current_config()
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        exit(1)
    
    # Ensure directories exist and print configuration
    ensure_directories()
    print_current_config()

    # Classification short-circuit (prepare once in main and reuse)
    if config.mode == "classification":
        run_classification_evaluation(config)
        return
    
    # Preprocess data
    bundle = prepare_data(
        ticker=config.data.company,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        target_feature=config.data.target_feature,
        lookback=config.data.lookback,
        horizon=config.data.horizon,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        target_as_return=config.data.target_as_return,
        use_log_returns=config.data.use_log_returns,
        scale=config.data.scale,
        cache_dir='cache',
        use_sentiment=config.data.use_sentiment,
        include_social=config.data.include_social,
        news_source='all',  # Options: 'all', 'google', 'yahoo', 'businesstoday'
    )

    print(f"\n{bundle.summary()}")

    # Generate meta path and inspection plots
    meta_path = generate_meta_path(config)
    generate_inspection_plots(config, bundle, meta_path)

    # Create and train/load model
    model = create_model(config, bundle, meta_path, args)

    # Generate predictions and evaluate
    run_regression_evaluation(model, bundle, args, meta_path)

if __name__ == "__main__":
    main()
