# File: train.py
# Authors: Kha Anh Nguyen :)
# Date: 24/08/2025

import pandas as pd
import os
from loguru import logger
from data_preprocessing.data_processor import DataProcessor
from utils.file_handling import save_data
from model.tf_models import TFModel
import tensorflow as tf

#------------------------------------------------------------------------------
# Train Model (load config from main.py)
#------------------------------------------------------------------------------
def train(cfg):
    """Train or load the model and return artifacts for downstream evaluation.

    Returns: (model, data, x_test, y_test, cfg)
    """
    # Resolve paths from cfg
    data_path = cfg.data_path
    model_path = cfg.model_path

    # Prepare data (load cache or process)
    if not os.path.exists(data_path):
        processor = DataProcessor(cache_dir='cache')
        data = processor.data_processing(
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            ticker=cfg.ticker,
            lag_days=cfg.lag_days,
            lookup_steps=cfg.lookup_steps,
            shuffle=cfg.shuffle,
            splitting_method=cfg.split_method,
            test_size=cfg.test_size,
            target_feature=cfg.target_feature,
            scale=cfg.scale,
            target_as_return=cfg.target_as_return,
            use_log_returns=cfg.use_log_returns,
        )
        if data:
            save_data(data, data_path)
        else:
            raise RuntimeError("Data processing failed")
    else:
        print("Loading cached processed data")
        data = pd.read_pickle(data_path)

    # Split prepared tensors
    x_train = data['X_train']
    y_train = data['y_train']
    x_test = data['X_test']
    y_test = data['y_test']

    print(f"Training data shape: x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"Test data shape: x_test={x_test.shape}, y_test={y_test.shape}")

    # Initialize model wrapper
    input_size = x_train.shape[2]
    lookup_steps = getattr(cfg, 'lookup_steps', 1)
    
    tf_model = TFModel(
        input_size=input_size,
        model_name=(cfg.model_name or 'lstm'),
        layers=cfg.layers,
        dropout_rate=cfg.dropout_rate,
        output_steps=lookup_steps,
    )

    # Train or load existing
    if not os.path.exists(model_path):
        print("Building and training model...")
        tf_model.fit(
            x_train,
            y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
        )
        
        # Generate training metrics chart
        if hasattr(tf_model, 'training_history'):
            from utils.plots import plot_training_metrics
            metrics_plot_path = model_path.replace('.keras', '_metrics.png')
            plot_training_metrics(tf_model.training_history, save_path=metrics_plot_path)
            print(f"Training metrics chart saved to: {metrics_plot_path}")

        
        tf_model.save_model(model_path)
        model = tf_model
    else:
        print("Using existing trained model")
        keras_model = tf.keras.models.load_model(model_path)  # type: ignore
        tf_model.model = keras_model
        model = tf_model

    return model, data, x_test, y_test, cfg