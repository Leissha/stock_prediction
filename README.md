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


#### Basic Usage (Default Settings)
```bash
# Navigate to dev directory first
cd dev

# Use default settings (CBA.AX, Close price, 1-day prediction)
python main.py

#### Target Feature Selection
```bash
# Predict different features (Close, Open, High, Low, Volume)
python main.py --company AAPL --target_feature Close --prediction_days 30
python main.py --company TSLA --target_feature Open --prediction_days 60
```


#### Different Split Methods
```bash
# Chronological split (default - recommended for time series)
python main.py --split_method date --test_size 0.2

# Random split (breaks temporal dependencies - use with caution)
python main.py --split_method random --test_size 0.2 --shuffle
```

#### Model Selection
```bash
# Standard LSTM (default)
python main.py --model_name lstm

# Bidirectional LSTM for potentially better performance
python main.py --model_name bilstm
```

#### Return-based Prediction
```bash
# Predict log returns instead of raw prices
python main.py --log_ret

# Predict simple percentage returns
python main.py --target_ret
```

#### More Configuration
```bash
# Complete example showcasing all 
python main.py \
    --company AAPL \
    --start_date 2022-01-01 \
    --end_date 2024-01-01 \
    --target_feature Close \
    --prediction_days 60 \
    --split_method date \
    --test_size 0.2 \
    --scale \
    --model_name lstm
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
- Multiple model architectures (LSTM, Bidirectional LSTM)
- Comprehensive trading-based evaluation metrics
- Outputs: Model files, plots, accuracy CSV files

---

## Command Line Arguments (dev module)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--company` | str | CBA.AX | Company ticker symbol |
| `--start_date` | str | 2 years ago | Start date for data |
| `--end_date` | str | Today | End date for data |
| `--target_feature` | str | Close | Target feature to predict (Close, Open, High, Low, AdjClose, Volume) |
| `--target_ret` | flag | False | Predict simple percentage returns instead of raw prices |
| `--log_ret` | flag | False | Predict log returns instead of raw prices |
| `--lag_days` | int | 60 | Number of days to look back for prediction |
| `--lookup_steps` | int | 1 | **Number of future days to predict (1=single-step, >1=multistep)** |
| `--test_size` | float | 0.2 | Test set size ratio |
| `--split_method` | str | 'date' | Split method (date, random) |
| `--shuffle` | flag | True | Shuffle data (for random split) |
| `--scale` | flag | True | Scale features |
| `--model_name` | str | lstm | Model type (lstm, gru, rnn, bilstm) |
| `--layers` | list | [50, 50, 50] | Layer sizes (e.g., 64 32 16) |
| `--dropout_rate` | float | 0.2 | Dropout rate for regularization |
| `--epochs` | int | 50 | Number of training epochs |
| `--batch_size` | int | 32 | Batch size for training |

## Output Files

- **Model**: `cache/trained_models/*.keras`
- **Plots**: `results/{some_config}.png`
- **Cache**: `cache/processed_data/*.pkl` and `cache/raw_data/*.pkl`
- **Results**: `results/{some_config}.csv`

## Project Structure
```
stock-prediction-project/
├── dev/                    # Advanced development module
│   ├── main.py             # CLI interface and orchestration
│   ├── train.py            # Model training script
│   ├── test.py             # Model evaluation and prediction
│   ├── model/              # AI Model architectures
│   │   └── tf_models.py    # TensorFlow models (LSTM, BiLSTM, GRU, RNN)
│   ├── data_preprocessing/ # Data processing modules
│   │   ├── data_loading.py # yfinance download + local cache
│   │   ├── data_processor.py # Main data processing pipeline
│   │   ├── data_splitting.py # Train/test split methods
│   │   ├── handle_nans.py  # NaN detection and interpolation
│   │   └── create_sequence.py # LSTM sliding window creation
│   ├── utils/              # Utilities (file_handling, eval, plots)
│   │   ├── file_handling.py # File I/O operations
│   │   ├── evaluating_utils.py # Evaluation metrics
│   │   └── plots.py        # Visualization functions
│   ├── config/             # Configuration files
│   │   ├── data.py         # Data configuration constants
│   │   └── run_config.py   # Runtime configuration dataclass
│   ├── cache/              # Cache processed data & models
│   │   ├── raw_data/       # Cached raw stock data
│   │   ├── processed_data/ # Cached processed sequences
│   │   ├── scalers/        # Cached feature scalers
│   │   └── trained_models/ # Saved model files
│   └── results/            # Output files (CSV & plots)
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
- ✅ **Task 2 Complete**: Data cleaning, code refactoring and modularization
- ✅ **Task 3 Complete**: Candlestick chart, return-based prediction, data leakage fixes
- ✅ **Task 4 Complete**: Feature engineering, trading metrics, comprehensive evaluation
- ✅ **Task 5 Complete**: Multivariate & multistep prediction implementation

---

**Last Updated**: September 5, 2025  
**Course**: COS30018 Option C
