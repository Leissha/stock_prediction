import os
from config.data import *
from argparse import ArgumentParser
from train import train
from config.run_config import RunConfig
from test import test_and_evaluate
from utils.evaluating_utils import trading_simulation

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
    
    # Multistep prediction arguments
    parser.add_argument("--lookup_steps", type=int, default=1,
                        help="Number of future days to predict (default: 1, >1 for multistep)")

    
    return parser.parse_args()

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    for d in ["cache","cache/trained_models","cache/processed_data","cache/raw_data","cache/scalers","results"]:
        os.makedirs(d, exist_ok=True)

    if args.log_ret:
        ret_suffix = "_log_ret"
    elif args.target_ret:
        ret_suffix = "_ret"
    else:
        ret_suffix = ""
    base_path = f"{args.company}_{args.start_date}_to_{args.end_date}_{args.target_feature}{ret_suffix}_step_{args.lookup_steps}"
    meta_path = f"{base_path}_seq-{args.lag_days}_{args.model_name}_layers{args.layers}_dropout{args.dropout_rate}_epochs{args.epochs}_bs{args.batch_size}"

    cfg = RunConfig(
        ticker=args.company,
        start_date=args.start_date,
        end_date=args.end_date,
        target_feature=args.target_feature,
        lag_days=args.lag_days,
        test_size=args.test_size,
        split_method=args.split_method,
        shuffle=args.shuffle,
        scale=args.scale,
        model_name=args.model_name,
        layers=args.layers,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_as_return=(args.target_ret or args.log_ret),
        use_log_returns=args.log_ret,
        base_path=base_path,
        data_path=f"cache/processed_data/{base_path}.pkl",
        model_path=f"cache/trained_models/{meta_path}.keras",
        plot_path=f"results/{meta_path}_predictions_chart.png",
        report_path=f"results/{meta_path}.csv",
        lookup_steps=args.lookup_steps,
    )

    print("="*60)
    print("STOCK PREDICTION PIPELINE")
    print("="*60)

    # Train the model
    model, data, x_test, y_test, cfg = train(cfg)

    # Test, evaluate, and plot the model
    res = test_and_evaluate(
        model=model,
        data=data,
        x_test=x_test,
        y_test=y_test,
        ticker=cfg.ticker,
        target_as_return=cfg.target_as_return,
        use_log_returns=cfg.use_log_returns,
        lag_days=cfg.lag_days,
        plot_path=cfg.plot_path,
    )
    
    # Calculate trading metrics
    metrics = trading_simulation(
        actual_prices=res['actual_prices'],
        predicted_prices=res['predicted_prices'],
        current_prices=res['new_current_prices'],
        lookup_step=cfg.lookup_steps,
        future_price=res['new_future_price'],
        loss_val=res['new_metrics']['loss'],
        mae_val=res['new_metrics']['mae'],
        rmse_val=res['new_metrics']['rmse'],
        filename=cfg.report_path,
        scale=cfg.scale,
        target_feature=data['target_feature'],
        training_features=data['training_features'],
    )

    print(f"Test results saved to: {cfg.report_path}")