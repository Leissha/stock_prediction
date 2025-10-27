"""
Financial AI Pipeline - Clean Main Entry Point

This module provides a clean, modular entry point for all 5 execution modes:
1. Sentiment Analysis
2. TF Models (LSTM, GRU, RNN, BiLSTM)  
3. SARIMAX
4. Classification
5. Ensemble
"""

import os
from argparse import ArgumentParser
from config.data import *
from config.pipeline_config import update_config_from_args, validate_current_config, print_current_config
from pipeline import prepare_data
from eval.regression_evaluator import run_regression_evaluation
from eval.classification_evaluator import run_classification_evaluation

def ensure_directories():
    """Ensure all required directories exist."""
    os.makedirs("results", exist_ok=True)
    os.makedirs("inspect_data", exist_ok=True)
    os.makedirs("cache/trained_models", exist_ok=True)

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
    
    # Execution mode arguments
    parser.add_argument("--classification", action="store_true", default=False,
                        help="Run binary classification (up/down) evaluation and exit")
    parser.add_argument("--model_name", type=str, default='lstm', 
                        help="Model type to use", 
                        choices=['lstm', 'gru', 'rnn', 'bilstm', 'sarimax', 'ensemble'])
    
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
    
    # SARIMAX arguments
    parser.add_argument("--sarimax_seasonal", action="store_true", default=False,
                        help="Enable seasonal SARIMAX")
    parser.add_argument("--sarimax_m", type=int, default=5,
                        help="Seasonal period m (default: 5)")
    
    # Ensemble arguments
    parser.add_argument("--ensemble_2", type=str, default='lstm',
                        help="Base TF model for ensemble (when --model_name ensemble)",
                        choices=['lstm', 'gru', 'rnn', 'bilstm'])
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

    base_path = f"{config.data.company}_{config.data.start_date}_to_{config.data.end_date}_{config.data.target_feature}{ret_suffix}_step_{config.data.horizon}"
    meta_path = f"{base_path}_seq-{config.data.lookback}_{meta_model_tag}_layers{config.tf_model.layers}_dropout{config.tf_model.dropout_rate}_epochs{config.tf_model.epochs}_bs{config.tf_model.batch_size}"
    
    return meta_path

def create_model(config, bundle, meta_path):
    """Create and train/load model based on configuration"""
    input_size = bundle.X_train.shape[2]
    
    if config.mode == "tf_model":
        return _create_tf_model(config, bundle, meta_path, input_size)
    elif config.mode == "sarimax":
        return _create_sarimax_model(config, bundle)
    elif config.mode == "ensemble":
        return _create_ensemble_model(config, bundle, meta_path)
    else:
        raise ValueError(f"Unknown execution mode: {config.mode}")

def _create_tf_model(config, bundle, meta_path, input_size):
    """Create and train/load TensorFlow model"""
    from model.tf_models import TFModel
    
    model = TFModel(
        input_size=input_size, 
        model_name=config.tf_model.model_name, 
        layers=config.tf_model.layers, 
        dropout_rate=config.tf_model.dropout_rate, 
        output_steps=config.data.horizon
    )

    # Simple path construction - much cleaner!
    model_path = f"cache/trained_models/{config.data.company}_{config.tf_model.model_name}_{meta_path.replace('.keras', '')}.keras"
    
    if not os.path.exists(model_path):
        print(f"Training {config.tf_model.model_name.upper()} model...")
        val_tuple = None
        if bundle.X_val is not None and bundle.y_val is not None:
            val_tuple = (bundle.X_val, bundle.y_val)
        
        model.fit(
            bundle.X_train,
            bundle.y_train,
            epochs=config.tf_model.epochs,
            batch_size=config.tf_model.batch_size,
            learning_rate=config.tf_model.learning_rate,
            optimizer=config.tf_model.optimizer,
            validation_data=val_tuple,
            meta_path=meta_path,
        )
        model.save_model(model_path)
    else:
        print(f"Loading existing model from {model_path}")
        from tensorflow import keras as tf_keras # type: ignore
        keras_model = tf_keras.models.load_model(model_path)
        model.model = keras_model
    
    return model

def _create_sarimax_model(config, bundle):
    """Create and train SARIMAX model"""
    from model.sarimax import SARIMAXModel
    
    print("Training SARIMAX model...")
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

def _create_ensemble_model(config, bundle, meta_path):
    """Create and train ensemble model"""
    from model.ensemble import EnsembleModel, EnsembleConfig
    
    ensemble_config = EnsembleConfig(
        sarima_weight=config.ensemble.sarima_weight, 
        model_2_weight=config.ensemble.model_2_weight
    )
    model = EnsembleModel(ensemble_config)
    
    # TF model cache path
    model_2_meta_path = f"{config.data.company}_{config.data.start_date}_to_{config.data.end_date}_{config.data.target_feature}_step_{config.data.horizon}_seq-{config.data.lookback}_{config.ensemble.ensemble_2}_layers{config.tf_model.layers}_dropout{config.tf_model.dropout_rate}_epochs{config.tf_model.epochs}_bs{config.tf_model.batch_size}"
    
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
        'learning_rate': config.tf_model.learning_rate 
    }
    
    # Fit both models
    model.fit_from_bundle(bundle, sarima_args, model_2, model_2_meta_path)
    return model

def generate_inspection_plots(config, bundle, meta_path):
    """Generate optional inspection plots"""
    if not hasattr(config, 'inspect_plots') or not config.inspect_plots:
        return
    
    try:
        from utils.plots import create_candlestick_chart, create_boxplot
        if bundle.train_df is not None:
            create_candlestick_chart(
                bundle.train_df, 
                config.data.company, 
                save_path=f"inspect_data/{meta_path}_candlestick.png", 
                n_days=1
            )
            create_boxplot(
                bundle.train_df, 
                config.data.company, 
                save_path=f"inspect_data/{meta_path}_boxplot.png", 
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
    
    # Create directories and print configuration
    ensure_directories()
    print_current_config()

    # Classification short-circuit
    if config.mode == "classification":
        run_classification_evaluation(
            ticker=config.data.company,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            use_sentiment=config.data.use_sentiment,
            scale=config.data.scale,
            test_size=config.data.test_size,
            val_size=config.data.val_size
        )
        return

    # Prepare data
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
    )

    print(f"\n{bundle.summary()}")

    # Generate meta path and inspection plots
    meta_path = generate_meta_path(config)
    generate_inspection_plots(config, bundle, meta_path)

    # Create and train/load model
    model = create_model(config, bundle, meta_path)

    # Generate predictions and evaluate
    run_regression_evaluation(model, bundle, args, meta_path)

if __name__ == "__main__":
    main()