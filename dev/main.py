"""
New main.py using pipeline wrapper with Pydantic DataBundle.
Keeps all argparse logic from old main.py.
"""
import os
import numpy as np
from argparse import ArgumentParser
from config.data import *
from pipeline import prepare_data, predict, predictions_to_prices
from eval.metrics import compute_metrics
from utils.plots import plot_predictions, create_candlestick_chart, create_boxplot, plot_training_metrics


def parse_args():
    """Parse command line arguments (same as old main.py)"""
    parser = ArgumentParser(description='Stock Price Prediction with LSTM')

    # Data loading arguments
    parser.add_argument("--company", type=str, default=TICKER,
                        help="Company ticker symbol (default: %(default)s)")
    parser.add_argument("--start_date", type=str, default=START_DATE,
                        help="Start date for data (default: %(default)s)")
    parser.add_argument("--end_date", type=str, default=END_DATE,
                        help="End date for data (default: %(default)s)")

    # Feature selection arguments
    parser.add_argument("--target_feature", type=str, default=TARGET_FEATURE,
                        help="Target feature to predict (default: Close)",
                        choices=["Close", "Open", "High", "Low", "AdjClose", "Volume"])
    parser.add_argument("--target_ret", action="store_true", default=False,
                        help="Predict simple percentage returns instead of raw prices")
    parser.add_argument("--log_ret", action="store_true", default=False,
                        help="Predict log returns instead of raw prices")
    parser.add_argument("--lookback", type=int, default=LOOKBACK,
                        help="Number of days to look back (default: %(default)s)")

    # Data processing arguments
    parser.add_argument("--test_size", type=float, default=TEST_SIZE,
                        help="Test size ratio (default: %(default)s)")
    parser.add_argument("--val_size", type=float, default=VAL_SIZE,
                        help="Validation size ratio (default: %(default)s)")
    parser.add_argument("--scale", action="store_true", default=SCALE,
                        help="Scale the data")

    # Model arguments
    parser.add_argument("--model_name", type=str, default='lstm', 
                        help="Model type to use", 
                        choices=['lstm', 'gru', 'rnn', 'bilstm', 'sarimax', 'ensemble'])
    parser.add_argument("--ensemble_2", type=str, default='lstm',
                        help="Base TF model for ensemble (when --ensemble_2)",
                        choices=['lstm', 'gru', 'rnn', 'bilstm'])
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
    parser.add_argument("--inspect_plots", action="store_true", default=False,
                        help="Generate preprocessing inspection plots (candlestick, boxplot)")

    # Multistep prediction
    parser.add_argument("--horizon", type=int, default=HORIZON,
                        help="Number of future days to predict (default: %(default)s)")

    # SARIMAX-specific arguments
    parser.add_argument("--sarimax_seasonal", action="store_true", default=False,
                        help="Enable seasonal SARIMAX")
    parser.add_argument("--sarimax_m", type=int, default=5,
                        help="Seasonal period m (default: 5)")

    # Ensemble-specific arguments
    parser.add_argument("--sarima_weight", type=float, default=0.5,
                        help="Weight for SARIMA in ensemble (default: 0.5)")
    parser.add_argument("--model_2_weight", type=float, default=0.5,
                        help="Weight for Model 2 in ensemble (default: 0.5)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create directories
    for d in ["cache", "cache/trained_models", "cache/processed_data", "cache/raw_data", "cache/scalers", "results"]:
        os.makedirs(d, exist_ok=True)

    # Determine target type suffix
    if args.log_ret:
        ret_suffix = "_log_ret"
    elif args.target_ret:
        ret_suffix = "_ret"
    else:
        ret_suffix = ""

    # Determine model tag
    if args.model_name == 'ensemble':
        meta_model_tag = f"ensemble+sarima-{args.sarima_weight}+{args.ensemble_2}-{args.model_2_weight}"
    else:
        meta_model_tag = f"{args.model_name}"

    base_path = f"{args.company}_{args.start_date}_to_{args.end_date}_{args.target_feature}{ret_suffix}_step_{args.horizon}"
    meta_path = f"{base_path}_seq-{args.lookback}_{meta_model_tag}_layers{args.layers}_dropout{args.dropout_rate}_epochs{args.epochs}_bs{args.batch_size}"

    print("=" * 60)
    print("STOCK PREDICTION PIPELINE")
    print("=" * 60)

    # ========================================================================
    # Step 1: Prepare data với Pydantic DataBundle
    # ========================================================================
    bundle = prepare_data(
        ticker=args.company,
        start_date=args.start_date,
        end_date=args.end_date,
        target_feature=args.target_feature,
        lookback=args.lookback,
        horizon=args.horizon,
        test_size=args.test_size,
        val_size=args.val_size,
        target_as_return=(args.target_ret or args.log_ret),
        use_log_returns=args.log_ret,
        scale=args.scale,
        cache_dir='cache',
    )

    print(f"\n{'=' * 60}")
    print(f"DATA SHAPES:")
    print(f"{'=' * 60}")
    print(f"X_train: {bundle.X_train.shape}")
    print(f"y_train: {bundle.y_train.shape}")
    if bundle.X_val is not None and bundle.y_val is not None:
        print(f"X_val: {bundle.X_val.shape}")
        print(f"y_val: {bundle.y_val.shape}")
    print(f"X_test: {bundle.X_test.shape}")
    print(f"y_test: {bundle.y_test.shape}")
    print(f"{'=' * 60}\n")

    # Optional: preprocessing inspection plots using the training dataframe
    if args.inspect_plots and bundle.train_df is not None:
        try:
            create_candlestick_chart(bundle.train_df, args.company, save_path=f"inspect_data/{meta_path}_candlestick.png", n_days=1)
            create_boxplot(bundle.train_df, args.company, save_path=f"inspect_data/{meta_path}_boxplot.png", n_days=20)
        except Exception:
            pass

    # ========================================================================
    # Step 2: Initialize model
    # ========================================================================
    input_size = bundle.X_train.shape[2]

    if args.model_name in ['lstm', 'gru', 'rnn', 'bilstm']:
        from model.tf_models import TFModel
        import tensorflow as tf

        model = TFModel(input_size=input_size, model_name=args.model_name, layers=args.layers, dropout_rate=args.dropout_rate, output_steps=args.horizon)

        model_path = f"cache/trained_models/{meta_path}.keras"
        if not os.path.exists(model_path):
            print(f"Training {args.model_name.upper()} model...")
            val_tuple = None
            if bundle.X_val is not None and bundle.y_val is not None:
                val_tuple = (bundle.X_val, bundle.y_val)
            model.fit(
                bundle.X_train,
                bundle.y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                optimizer=args.optimizer,
                validation_data=val_tuple,
                meta_path=meta_path,
            )
            model.save_model(model_path)
        else:
            print(f"Loading existing model from {model_path}")
            from tensorflow import keras as tf_keras  # type: ignore
            keras_model = tf_keras.models.load_model(model_path)
            model.model = keras_model

    elif args.model_name == 'sarimax':
        from model.sarimax import SARIMAXModel

        print("Training SARIMAX model...")
        model = SARIMAXModel(
            seasonal=args.sarimax_seasonal,
            m=args.sarimax_m,
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

    elif args.model_name == 'ensemble':
        # Ensemble builds its own submodels from the bundle
        from model.ensemble import EnsembleModel, EnsembleConfig
        config = EnsembleConfig(sarima_weight=args.sarima_weight, model_2_weight=args.model_2_weight)
        model = EnsembleModel(config)
        
        # TF model cache path
        model_2_meta_path = f"{base_path}_seq-{args.lookback}_{args.ensemble_2}_layers{args.layers}_dropout{args.dropout_rate}_epochs{args.epochs}_bs{args.batch_size}"
        # Extract only related args
        sarima_args = { 'seasonal': args.sarimax_seasonal, 'm': args.sarimax_m }
        model_2 = { 'layers': args.layers, 'dropout_rate': args.dropout_rate, 'epochs': args.epochs, 'batch_size': args.batch_size, 'model_name': args.ensemble_2, 'optimizer': args.optimizer, 'learning_rate': args.learning_rate }
        
        # Fit both models
        model.fit_from_bundle(bundle, sarima_args, model_2, model_2_meta_path)
    else:
        raise ValueError(f"Unknown model: {args.model_name}")

    # ========================================================================
    # Step 3: Predict
    # ========================================================================
    print(f"\nGenerating predictions...")
    y_hat = predict(model, bundle)
    y_true = bundle.y_test

    print(f"Predictions shape: {y_hat.shape}")
    print(f"Ground truth shape: {y_true.shape}")

    # ========================================================================
    # Step 4: Convert to price space
    # ========================================================================
    print(f"\nConverting to price space...")
    y_true_px, y_hat_px = predictions_to_prices(y_true, y_hat, bundle)

    print(f"Price space - y_true: [{y_true_px.min():.2f}, {y_true_px.max():.2f}]")
    print(f"Price space - y_hat: [{y_hat_px.min():.2f}, {y_hat_px.max():.2f}]")

    # ========================================================================
    # Step 5: Compute metrics
    # ========================================================================
    # Extract epochs_trained if available (for TF models)
    epochs_trained = getattr(model, 'epochs_trained', None)
    if epochs_trained is None:
        # For ensemble, try to get from the TF submodel
        model_2 = getattr(model, 'model_2', None)
        if model_2 is not None:
            epochs_trained = getattr(model_2, 'epochs_trained', None)

    metrics = compute_metrics(y_true_px, y_hat_px, epochs_trained=epochs_trained)

    # Plot prediction vs actual (first step in horizon)
    # Build real date axis from test_df index if available
    dates = None
    if hasattr(bundle, 'test_df') and bundle.test_df is not None:
        try:
            dates = bundle.test_df.index[:len(y_true_px)]
        except Exception:
            dates = None
    # pass meta and dates positionally to match signature (actual, predicted, ticker, meta=None, dates=None)
    plot_predictions(y_true_px[:, 0], y_hat_px[:, 0], args.company, f"results/{meta_path}_predictions.png", dates)

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS - {args.model_name.upper()}")
    print(f"{'=' * 60}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Directional Accuracy (DA@1): {metrics['directional_accuracy']:.3f}")
    if 'epochs_trained' in metrics:
        print(f"Epochs Trained: {metrics['epochs_trained']}/{args.epochs}")
    print(f"{'=' * 60}")

    # Save results
    import pandas as pd
    metric_names = ['mae', 'rmse', 'directional_accuracy']
    metric_values = [metrics['mae'], metrics['rmse'], metrics['directional_accuracy']]

    # Add epochs_trained if present
    if 'epochs_trained' in metrics:
        metric_names.append('epochs_trained')
        metric_values.append(metrics['epochs_trained'])

    results_df = pd.DataFrame({
        'metric': metric_names,
        'value': metric_values
    })
    report_path = f"results/{meta_path}.csv"
    results_df.to_csv(report_path, index=False)
    print(f"\nResults saved to: {report_path}")

    # Final status
    print(f"\n{'=' * 60}")
    if np.isfinite(y_hat_px).all() and (y_hat_px > 0).all():
        print("STATUS: OK")
    else:
        print("STATUS: FAIL (NaN/Inf/negative prices detected)")
    print(f"{'=' * 60}")
