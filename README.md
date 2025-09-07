# Stock Prediction Project - COS30018 Option C

## Project Overview
This project demonstrates a stock prediction system and comparing multiple AI Models approaches for stock forecasting
## Quick Start

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage Examples

---



### 🚀 Quick Start
```bash
# PyTorch backend
python dev/train.py --model_name lstm --backend pytorch

# TensorFlow backend
python dev/train.py --model_name bilstm --backend tensorflow

# Autoload overall best config (if present)
python dev/train.py --backend tensorflow

# Custom hyperparameters
python dev/train.py --model_name lstm --layers "128,64,32" --dropout_rate 0.3 --backend pytorch
```

### 📊 Results & Artifacts
- Trained models: `dev/cache/trained_models/*.h5` (TF) or `*.pth` (PyTorch)
- Processed data cache: `dev/cache/processed_data/*.pkl`
- Results CSV: `dev/results/{config}.csv`
- Prediction chart: `dev/results/{config}_predictions_chart.png`

---

### 🔧 **dev - Current Development Module (Task 2 Complete)**
**Purpose**: Data processing with functionalities:
1. Cache data
2. Split choices (random, ratio, date)
3. Handle NaNs
4. Optional scaling
5. More features with args...

#### Basic Usage (Default Settings)
```bash
# Use default settings (CBA.AX, Close price, 30 days)
python dev/train.py
```

#### Target Feature Selection
```bash
# Predict different features (Close, Open, High, Low, Volume)
python dev/train.py --company AAPL --target_feature Close --prediction_days 30
python dev/train.py --company TSLA --target_feature Open --prediction_days 60
```


#### Different Split Methods
```bash
# Chronological split (default - recommended for time series)
python dev/train.py --split_method date --test_size 0.2

# Random split (breaks temporal dependencies - use with caution)
python dev/train.py --split_method random --test_size 0.2 --shuffle
```

#### Model Selection
Use `train.py` with `--backend pytorch|tensorflow` and `--model_name lstm|gru|rnn|bilstm`.

#### More Configuration
```bash
# Complete example showcasing all 
python dev/train.py \
    --company AAPL \
    --start_date 2022-01-01 \
    --end_date 2024-01-01 \
    --target_feature Close \
    --prediction_days 60 \
    --split_method date \
    --test_size 0.2 \
    --scale \
    --model_name lstm
# Task 2: Date range control, NaN handling, caching, separate scalers per feature
```

**What it does:**
- **Multi-feature input**: Uses all OHLCV features automatically
- **Flexible target selection**: Predict any feature (Close, Open, High, Low, Volume)  
- **Date range control**: Specify exact start/end dates for data
- **Smart NaN handling**: Linear interpolation for missing stock data
- **Multiple split methods**: Chronological, ratio, or random data splitting
- **Advanced caching**: Raw data + processed data + scaler persistence
- **Separate scalers**: Industry best practice for OHLCV features
- **Data leakage prevention**: Scalers fit only on training data
- **Data visualization**: Interactive candlestick charts with SMA/EMA and boxplots
- **Flexible model builder**: Dynamic DL model construction with multiple architectures
- **Multiple model architectures**: LSTM, GRU, RNN, Bidirectional LSTM, Dense
- **Comprehensive experimentation**: Framework for testing different hyperparameter configurations
- **Advanced training**: Early stopping, learning rate scheduling, callbacks
- Comprehensive trading-based evaluation metrics
- Outputs: Model files, plots, accuracy CSV files, inspection charts, and experiment results

---

## Command Line Arguments (dev/train.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--company` | str | CBA.AX | Company ticker symbol |
| `--start_date` | str | 2 years ago | Start date for data |
| `--end_date` | str | Today | End date for data |
| `--target_feature` | str | Close | Target to predict (Close, Open, High, Low, AdjClose, Volume) |
| `--lag_days` | int | 60 | Number of lookback days for sequences (look back)|
| `--test_size` | float | 0.2 | Test set size ratio |
| `--split_method` | str | 'date' | Split method (date, ratio, random) |
| `--shuffle` | flag | False | Shuffle data (used with random split) |
| `--scale` | flag | True | Scale features |
| `--model_name` | str | None | Model type (lstm, gru, rnn, bilstm) |
| `--layers` | str | "64,32" | Layer sizes (comma-separated) |
| `--dropout_rate` | float | 0.2 | Dropout rate |
| `--learning_rate` | float | 0.001 | Learning rate |
| `--optimizer` | str | adam | Optimizer (adam, rmsprop, sgd) |
| `--epochs` | int | 25 | Training epochs |
| `--batch_size` | int | 32 | Training batch size |
| `--backend` | str | pytorch | Backend (pytorch, tensorflow) |


## Output Files

- **Trained Models**: `dev/cache/trained_models/*.h5` (TF) or `*.pth` (PyTorch)
- **Prediction Plots**: `dev/results/{config}_predictions_chart.png`
- **Results CSV**: `dev/results/{config}.csv`
- **Inspection Charts**: `dev/cache/inspect_data/{ticker}_{date_range}/`
  - `candlestick_chart.png`, `boxplot.png`
- **Cache**: `dev/cache/processed_data/*.pkl`, `dev/cache/raw_data/*.pkl`, `dev/cache/scalers/*.pkl`

## Project Structure
```
stock-prediction-project/
├── dev/                    # Advanced development module
│   ├── train.py            # Main hybrid training script (PyTorch/TF)
│   ├── model/              # Model builders
│   │   ├── pytorch_models.py   # PyTorch LSTM/GRU/RNN/BiLSTM
│   │   └── tf_models.py         # TensorFlow LSTM/GRU/RNN/BiLSTM
│   ├── data_preprocessing/ # Data processing modules
│   ├── cache/              # Cache directory
│   │   ├── trained_models/ # Trained models cache
│   │   ├── processed_data/ # Processed data cache
│   │   ├── raw_data/       # Raw data cache
│   │   ├── scalers/        # Scaler cache
│   │   └── finetune/       # Best config JSONs from Optuna
│   │   └── inspect_data/   # Data inspection charts
│   ├── experiment_results/ # Experiment results (Task C.4)
│   ├── config/             # Configuration files
│   └── results/            # Output files (accuracy.csv & plots)
├── utils/                  # Shared utilities (file_handling, eval, plots)
├── requirements.txt        # Package dependencies
└── README.md               # This file
```

## Requirements
| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **Core ML** | tensorflow | ≥2.19.0 | Deep learning framework |
| | keras | ≥3.11.1 | High-level neural network API |
| | scikit-learn | ≥1.7.1 | Machine learning utilities |
| | numpy | ≥2.1.3 | Numerical computing |
| | pandas | ≥2.3.1 | Data manipulation |
| **Visualization** | matplotlib | ≥3.10.5 | Plotting library |
| | plotly | ≥5.17.0 | Interactive charts |
| | kaleido | ≥0.2.1 | Static image export |
| **Data Fetching** | yfinance | ≥0.2.65 | Yahoo Finance data |
| **Utilities** | requests | ≥2.32.4 | HTTP library |
| | loguru | ≥0.7.0 | Logging |

## Troubleshooting

### Common Issues
1. **Import Errors**: Make sure virtual environment is activated
2. **Data Download Issues**: Check internet connection and yfinance availability
3. **Memory Issues**: Reduce `prediction_days` or use smaller datasets
4. **Model Training**: Ensure sufficient data for the specified lookback period

### Getting Help
- Check the logs for detailed error messages
- Verify all dependencies are installed correctly
- Ensure you have sufficient disk space for caching

## Status
- ✅ **Task 1 Complete**: Environment setup, code testing, performance comparison
- ✅ **Task 2 Complete**: Data cleaning, code refactoring and modularization (for future extensibility & modifiablility :)
- ✅ **Task 3 Complete**: Data visualization with candlestick charts and boxplots
- ✅ **Task 4 Complete**: Flexible Deep Learning model builder with multiple architectures (LSTM, GRU, RNN, Bidirectional LSTM)

---

**Last Updated**: August 28, 2025  
**Course**: COS30018 Option C
