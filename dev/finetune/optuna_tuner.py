"""
Simple PyTorch Optuna Tuner
Finds best hyperparameters for stock prediction models
"""

import optuna
import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
import argparse
from loguru import logger
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.data_processor import DataProcessor
from data_preprocessing.data_splitting import val_split_with_prices
from model.pytorch_models import pytorch_model_factory
from train.pytorch_trainer import fit, predict, evaluate, get_device
from config.data import COMPANY, TRAIN_START, TRAIN_END, LAG_DAYS, TEST_SIZE, SCALE, SHUFFLE, SPLIT_METHOD, PRICE_VALUE
from utils.file_handling import init_cache_dir
from utils.evaluating_utils import calculate_trading_metrics_returns


def setup_gpu():
    """Simple GPU setup."""
    device = get_device()
    gpu_available = str(device).startswith('cuda')
    if gpu_available:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")
    return device, gpu_available


def objective(trial, model_name, device, x_train, y_train, x_val, y_val, current_prices_val, target_scaler):
    """Objective function for Optuna optimization."""
    
    # Hyperparameters
    n_layers = trial.suggest_int('n_layers', 2, 6)  
    layers = [trial.suggest_int(f'units_l{i}', 32, 256) for i in range(n_layers)] 
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 50, 200)  # Tighter bounds for faster optimization
    
    # Trading parameters
    threshold = trial.suggest_float('threshold', 0.005, 0.05)  # Wider range to avoid always-in-market
    cost = trial.suggest_float('cost', 0.0002, 0.002, log=True)  # 2-20 bps transaction cost
    
    try:
        # Create model
        model = pytorch_model_factory(
            input_size=x_train.shape[2],
            model_name=model_name,
            layers=layers,
            dropout_rate=dropout_rate
        )
        
        # Train model
        start_time = time.time()
        model = fit(model, x_train, y_train, x_val, y_val, 
                   epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=20, device=device)
        training_time = time.time() - start_time
        
        # Evaluate using trading metrics with proper price alignment
        predictions = predict(model, x_val, device)

        # Sanity check predictions (debug flat-line issues)
        pred_std = float(np.std(predictions))
        if pred_std < 1e-3:
            logger.warning(f"Trial {trial.number}: Low prediction variance (std={pred_std:.6f}) - model may be collapsing")

        # Inverse-scale predictions and y_val to price level; current_prices are already in price units
        if target_scaler is not None:
            pred_prices = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
            y_val_prices = target_scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
        else:
            pred_prices = predictions.flatten()
            y_val_prices = y_val.flatten()
        curr_prices = current_prices_val.flatten()
        
        # Calculate returns (aligned)
        actual_returns = (y_val_prices - curr_prices) / curr_prices
        predicted_returns = (pred_prices - curr_prices) / curr_prices
        
        # Use consistent clean evaluation (Rule B, C)
        from utils.evaluating_utils import trading_metrics_from_prices
        
        # Create price sequences 
        p_true_seq = np.concatenate([curr_prices[:1], y_val_prices])
        p_pred_seq = np.concatenate([curr_prices[:1], pred_prices])
        
        trading_metrics = trading_metrics_from_prices(
            p_true_seq, p_pred_seq, threshold=threshold, cost=cost
        )

        # Quick diagnostics for first trial to detect issues (units, always-in-market)
        if trial.number == 0:
            signals = trading_metrics['signals']
            pnl = trading_metrics['pnl_pct']
            dir_acc = float((np.sign(predicted_returns) == np.sign(actual_returns)).mean())
            frac_long = float((signals == 1).mean())
            frac_short = float((signals == -1).mean())  
            frac_hold = float((signals == 0).mean())
            logger.info(
                f"Diagnostics: dir_acc={dir_acc:.3f}, price_mean={np.mean(curr_prices):.2f}, "
                f"pred_ret_std={np.std(predicted_returns):.4f}, signals +1/0/-1= {frac_long:.2f}/{frac_hold:.2f}/{frac_short:.2f}, "
                f"profit_per_trade={trading_metrics['profit_per_trade_pct']:.4f}"
            )
        
        # Optimize for percentage profit directly (maximize)
        score = trading_metrics['total_profit_pct']
        
        logger.info(f"Trial {trial.number}: Score={score:.4f}, Profit%={score:.4f}, "
                   f"Accuracy={trading_metrics['accuracy']:.3f}, Trades={trading_metrics['total_trades']}, "
                   f"Time={training_time:.1f}s")
        
        return score
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return float('-inf')


def run_optuna_for_model(model_name, n_trials=30):
    """Run Optuna optimization for a specific model type."""
    logger.info(f"Starting optimization for {model_name.upper()} model ({n_trials} trials)")
    
    # Setup GPU
    device, gpu_available = setup_gpu()
    
    # Load data
    logger.info("Loading data...")
    processor = DataProcessor(cache_dir='cache')
    
    # Use same data period as main training for consistency
    data = processor.data_processing(
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        ticker=COMPANY,
        lag_days=LAG_DAYS,
        lookup_step=1,
        shuffle=SHUFFLE,
        splitting_method=SPLIT_METHOD,
        test_size=TEST_SIZE,
        target_feature=PRICE_VALUE,
        scale=SCALE
    )
    
    if not data:
        logger.error("Data processing failed")
        return None
    
    # Extract data
    x_train = data['X_train']
    y_train = data['y_train']
    current_prices_train = data['current_prices_train']
    # Use dedicated target scaler for inverse-scaling to price
    target_feature = data['target_feature'].lower()
    target_key = f"{COMPANY}_{target_feature}_target"
    target_scaler = data['scalers'].get(target_key)
    
    # Get feature scaler for current prices
    feature_key = f"{COMPANY}_{target_feature}"
    feature_scaler = data['scalers'].get(feature_key)
    
    # Create validation split (20% of training data) with current prices
    x_train, y_train, x_val, y_val, current_prices_train, current_prices_val = val_split_with_prices(
        x_train, y_train, current_prices_train, val_size=0.2
    )
    
    # Fix current_prices_val using feature scaler if needed
    if current_prices_val is None and feature_scaler is not None:
        close_idx = 0  # 'close' is index 0
        curr_close_scaled = x_val[:, -1, close_idx].reshape(-1, 1)
        current_prices_val = feature_scaler.inverse_transform(curr_close_scaled).reshape(-1)
        logger.info("Reconstructed validation current prices using feature scaler")
    
    # Create study with pruning for faster optimization (maximize)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # Optimize
    start_time = time.time()
    study.optimize(
        lambda trial: objective(trial, model_name, device, x_train, y_train, x_val, y_val, current_prices_val, target_scaler),
        n_trials=n_trials,
        show_progress_bar=True
    )
    total_time = time.time() - start_time
    
    # Results
    logger.info(f"Optimization completed in {total_time:.1f}s")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best score: {study.best_trial.value:.4f}")
    
    # Save results
    results = {
        'model_name': model_name,
        'best_hyperparameters': study.best_trial.params,
        'best_score': study.best_trial.value,
        'best_trial_number': study.best_trial.number,
        'total_trials': len(study.trials),
        'total_time': total_time,
        'gpu_used': gpu_available
    }
    
    # Save to file
    os.makedirs('cache/finetune', exist_ok=True)
    output_file = f"cache/finetune/best_{model_name}_pytorch_config.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: cache/finetune/best_{model_name}_pytorch_config.json")
    return results


def main():
    """Main function to run Optuna optimization."""
    parser = argparse.ArgumentParser(description='Simple PyTorch Optuna Tuner')
    parser.add_argument('--models', nargs='+', default=['lstm', 'gru', 'rnn', 'bilstm'],
                       help='Model types to optimize (default: lstm gru rnn bilstm)')
    parser.add_argument('--trials', type=int, default=30,
                       help='Number of trials per model (default: 30)')

    args = parser.parse_args()
    
    init_cache_dir()
    
    logger.info("="*60)
    logger.info("SIMPLE PYTORCH OPTUNA TUNER")
    logger.info("="*60)
    
    all_results = {}
    
    for model_name in args.models:
        logger.info(f"Optimizing {model_name.upper()} model...")
        try:
            results = run_optuna_for_model(
                model_name=model_name,
                n_trials=args.trials
            )
            if results:
                all_results[model_name] = results
        except Exception as e:
            logger.error(f"Failed to optimize {model_name}: {e}")
    
    # Find overall best
    if all_results:
        best_model = min(all_results.keys(), key=lambda x: all_results[x]['best_score'])
        best_score = all_results[best_model]['best_score']
        
        logger.info(f"Overall Best Model: {best_model.upper()}")
        logger.info(f"Best Score: {best_score:.4f}")
        
        # Save overall best
        overall_results = {
            'best_model_name': best_model,
            'best_score': best_score,
            'best_hyperparameters': all_results[best_model]['best_hyperparameters'],
            'all_results': all_results
        }
        
        with open('cache/finetune/best_overall_pytorch_config.json', 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        logger.info("Overall best results saved to: cache/finetune/best_overall_pytorch_config.json")
    
    logger.info("="*60)
    logger.info("OPTUNA OPTIMIZATION COMPLETED")
    logger.info("="*60)


if __name__ == "__main__":
    main()
