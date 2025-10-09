"""
New main.py using pipeline wrapper with Pydantic DataBundle.
Keeps all argparse logic from old main.py.
"""
import os
import numpy as np
from argparse import ArgumentParser
from config.data import *
from pipeline import prepare_data, predict, to_price_space
from eval.metrics import compute_metrics


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
    parser.add_argument("--lag_days", type=int, default=LAG_DAYS,
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
    parser.add_argument("--layers", nargs='+', type=int, default=LAYERS,
                        help="Layer sizes separated by space (e.g: 64 32 16)")
    parser.add_argument("--dropout_rate", type=float, default=DROPOUT,
                        help="Dropout rate (default: 0.2)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size (default: 32)")

    # Multistep prediction
    parser.add_argument("--lookup_steps", type=int, default=1,
                        help="Number of future days to predict (default: 1)")

    # SARIMAX-specific arguments
    parser.add_argument("--sarimax_seasonal", action="store_true", default=False,
                        help="Enable seasonal SARIMAX")
    parser.add_argument("--sarimax_m", type=int, default=5,
                        help="Seasonal period m (default: 5)")

    # Ensemble-specific arguments
    parser.add_argument("--sarima_weight", type=float, default=0.5,
                        help="Weight for SARIMA in ensemble (default: 0.5)")
    parser.add_argument("--lstm_weight", type=float, default=0.5,
                        help="Weight for LSTM in ensemble (default: 0.5)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create directories
    for d in ["cache", "cache/trained_models", "cache/processed_data",
              "cache/raw_data", "cache/scalers", "results"]:
        os.makedirs(d, exist_ok=True)

    # Determine target type suffix
    if args.log_ret:
        ret_suffix = "_log_ret"
    elif args.target_ret:
        ret_suffix = "_ret"
    else:
        ret_suffix = ""

    base_path = f"{args.company}_{args.start_date}_to_{args.end_date}_{args.target_feature}{ret_suffix}_step_{args.lookup_steps}"
    meta_path = f"{base_path}_seq-{args.lag_days}_{args.model_name}_layers{args.layers}_dropout{args.dropout_rate}_epochs{args.epochs}_bs{args.batch_size}"

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
        lag_days=args.lag_days,
        lookup_steps=args.lookup_steps,
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

    # ========================================================================
    # Step 2: Initialize model
    # ========================================================================
    input_size = bundle.X_train.shape[2]
    lookup_steps = bundle.horizon

    if args.model_name in ['lstm', 'gru', 'rnn', 'bilstm']:
        from model.tf_models import TFModel
        import tensorflow as tf

        model = TFModel(
            input_size=input_size,
            model_name=args.model_name,
            layers=args.layers,
            dropout_rate=args.dropout_rate,
            output_steps=lookup_steps,
        )

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
                validation_data=val_tuple,
            )
            model.save_model(model_path)
            print(f"Model saved to {model_path}")
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
        config = EnsembleConfig(sarima_weight=args.sarima_weight, lstm_weight=args.lstm_weight)
        model = EnsembleModel(config)
        # LSTM cache path (independent of ensemble meta_path)
        lstm_base_path = f"{args.company}_{args.start_date}_to_{args.end_date}_{args.target_feature}{ret_suffix}_step_{args.lookup_steps}"
        lstm_meta_path = f"{lstm_base_path}_seq-{args.lag_days}_lstm_layers{args.layers}_dropout{args.dropout_rate}_epochs{args.epochs}_bs{args.batch_size}"
        lstm_model_path = f"cache/trained_models/{lstm_meta_path}.keras"
        sarima_args = { 'seasonal': args.sarimax_seasonal, 'm': args.sarimax_m }
        lstm_args = { 'layers': args.layers, 'dropout_rate': args.dropout_rate, 'epochs': args.epochs, 'batch_size': args.batch_size, 'model_name': 'lstm' }
        model.fit_from_bundle(bundle, sarima_args, lstm_args, lstm_model_path)

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
    y_true_px, y_hat_px = to_price_space(y_true, y_hat, bundle)

    print(f"Price space - y_true: [{y_true_px.min():.2f}, {y_true_px.max():.2f}]")
    print(f"Price space - y_hat: [{y_hat_px.min():.2f}, {y_hat_px.max():.2f}]")

    # ========================================================================
    # Step 5: Compute metrics
    # ========================================================================
    metrics = compute_metrics(y_true_px, y_hat_px)

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS - {args.model_name.upper()}")
    print(f"{'=' * 60}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Directional Accuracy (DA@1): {metrics['directional_accuracy']:.3f}")
    print(f"{'=' * 60}")

    # Save results
    import pandas as pd
    results_df = pd.DataFrame({
        'metric': ['mae', 'rmse', 'directional_accuracy'],
        'value': [metrics['mae'], metrics['rmse'], metrics['directional_accuracy']]
    })
    report_path = f"results/{meta_path}.csv"
    results_df.to_csv(report_path, index=False)
    print(f"\n✅ Results saved to: {report_path}")

    # Final status
    print(f"\n{'=' * 60}")
    if np.isfinite(y_hat_px).all() and (y_hat_px > 0).all():
        print("STATUS: OK ✅")
    else:
        print("STATUS: FAIL ❌ (NaN/Inf/negative prices detected)")
    print(f"{'=' * 60}")
