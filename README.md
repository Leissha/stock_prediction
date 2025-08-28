# Stock Prediction Project - COS30018 Option C

## Project Overview
This project implements a production-ready stock prediction system with industry best practices for financial time series forecasting.

### Key Highlights (Task 2 Complete)
- ✅ **Multi-Feature Support**: Uses all OHLCV features as input, predicts any target feature
- ✅ **Smart Data Processing**: Automatic date control, NaN handling, multiple split methods
- ✅ **Industry Best Practices**: Separate scalers per feature, data leakage prevention
- ✅ **Advanced Caching**: Raw data, processed data, and scaler persistence
- ✅ **Production Ready**: Comprehensive evaluation metrics and modular architecture

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

### 🎯 **Stock Prediction Pipeline (Task 2 Complete)**
**Purpose**: Production-ready financial time series forecasting system

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

#### Multi-Feature Input with Single Target ✅ NEW
```bash
# Uses ALL OHLCV features as input, predicts Close price
python dev/train.py --company AAPL --target_feature Close --prediction_days 60
# Automatic: Uses ['close', 'high', 'low', 'open', 'volume'] as input features
```

#### Different Split Methods ✅ NEW
```bash
# Chronological split (default - recommended for time series)
python dev/train.py --split_method date --test_size 0.2

# Random split (breaks temporal dependencies - use with caution)
python dev/train.py --split_method random --test_size 0.2 --shuffle
```

#### Model Selection
```bash
# Standard LSTM (default)
python dev/train.py --model_name lstm

# Bidirectional LSTM for potentially better performance
python dev/train.py --model_name bidirectional_lstm
```

#### Advanced Configuration with Task 2 Features
```bash
# Complete example showcasing all Task 2 requirements
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
# ✅ Automatic: Date range control, NaN handling, caching, separate scalers per feature
```

**What it does (Task 2 Enhanced):**
- ✅ **Multi-feature input**: Uses all OHLCV features automatically
- ✅ **Flexible target selection**: Predict any feature (Close, Open, High, Low, Volume)  
- ✅ **Date range control**: Specify exact start/end dates for data
- ✅ **Smart NaN handling**: Linear interpolation for missing stock data
- ✅ **Multiple split methods**: Chronological, ratio, or random data splitting
- ✅ **Advanced caching**: Raw data + processed data + scaler persistence
- ✅ **Separate scalers**: Industry best practice for OHLCV features
- ✅ **Data leakage prevention**: Scalers fit only on training data
- Multiple model architectures (LSTM, Bidirectional LSTM)
- Comprehensive trading-based evaluation metrics
- Outputs: Model files, plots, accuracy CSV files

---

## Command Line Arguments (dev module - Task 2 Enhanced)

| Argument | Type | Default | Description | Task 2 Requirement |
|----------|------|---------|-------------|-------------------|
| `--company` | str | CBA.AX | Company ticker symbol | |
| `--start_date` | str | 2023-08-27 | Start date for dataset | ✅ **1a** Date control |
| `--end_date` | str | Today | End date for dataset | ✅ **1a** Date control |
| `--target_feature` | str | 'Close' | Target to predict (Close, Open, High, Low, Volume) | ✅ Multi-feature |
| `--prediction_days` | int | 30 | Number of days to look back (LSTM sequence length) | |
| `--test_size` | float | 0.2 | Test set size ratio | |
| `--split_method` | str | 'date' | Split method (date, ratio, random) | ✅ **1c** Multiple splits |
| `--shuffle` | flag | False | Shuffle training sequences | |
| `--scale` | flag | True | Apply feature scaling with separate scalers | ✅ **1e** Scaler storage |
| `--model_name` | str | 'lstm' | Model type (lstm, bidirectional_lstm) | |

**Automatic Features (No Arguments Needed):**
- ✅ **Multi-feature input**: Always uses all available OHLCV features
- ✅ **NaN handling**: Automatic linear interpolation (**1b**)
- ✅ **Data caching**: Raw data + processed data persistence (**1d**)
- ✅ **Scaler caching**: Separate scalers saved to `cache/scalers/` (**1e**)

## Output Files

### Model Outputs
- **Trained Model**: `dev/cache/trained_models/{config}.h5`
- **Performance Plots**: `dev/results/{config}.png`
- **Trading Metrics**: `dev/results/{config}.csv`

### Cache Outputs (Task 2 Requirements)
- **Raw Data Cache**: `dev/cache/raw_data/{ticker}_{start}_to_{end}.pkl` ✅ **1d**
- **Processed Cache**: `dev/cache/processed_data/{config}.pkl` ✅ **1d**  
- **Scaler Cache**: `dev/cache/scalers/{ticker}_scalers.pkl` ✅ **1e**

### Example Output Naming
```
2023-08-27_CBA.AX_Close_seq-30-step_1_lstm.h5    # Model
2023-08-27_CBA.AX_Close_seq-30-step_1_lstm.png   # Plot
2023-08-27_CBA.AX_Close_seq-30-step_1_lstm.csv   # Metrics
```

## Project Structure (Cleaned)
```
stock-prediction-project/
├── dev/                    # Main prediction pipeline (Task 2 Complete)
│   ├── train.py            # Main training script
│   ├── cache/              # All caching (raw_data, processed_data, scalers, trained_models)
│   ├── config/             # Configuration files  
│   ├── data_preprocessing/ # Modular data processing pipeline
│   │   ├── data_processor.py    # Main processing orchestrator
│   │   ├── data_loading.py      # Stock data fetching & caching
│   │   ├── handle_nans.py       # Missing value interpolation
│   │   ├── data_splitting.py    # Train/test split methods
│   │   └── create_sequence.py   # LSTM sequence generation
│   ├── model/              # AI Model architectures
│   │   ├── lstm.py         # Standard LSTM implementation
│   │   └── bidirectional_lstm.py # Bidirectional LSTM
│   ├── utils/              # Evaluation and visualization utilities
│   │   ├── evaluating_utils.py  # Trading metrics calculation
│   │   ├── plots.py            # Prediction visualization
│   │   └── file_handling.py    # Generic file I/O operations
│   └── results/            # Output plots and CSV metrics
├── docs/                   # Documentation and reports
│   └── C.2/               # Task 2 documentation
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
| | jupyter | ≥1.0.0 | Notebook interface |
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

## Task 2 Features Implemented ✅

| Requirement | Status | Implementation |
|------------|--------|----------------|
| **1a** - Date specification | ✅ | `--start_date` and `--end_date` parameters |
| **1b** - NaN handling | ✅ | Linear interpolation in `handle_nans.py` |  
| **1c** - Multiple split methods | ✅ | `date`, `ratio`, `random` in `data_splitting.py` |
| **1d** - Local data caching | ✅ | Raw data + processed data + scaler persistence |
| **1e** - Scaler storage | ✅ | Separate scalers per feature, auto-saved |

## Status
- ✅ **Task 1 Complete**: Environment setup and baseline implementation
- ✅ **Task 2 Complete**: Advanced data processing with industry best practices
- 🚧 **Task 3 in progress**: Advanced visualization and analysis

---

**Last Updated**: August 29, 2025  
**Course**: COS30018 Option C  
**Pipeline Status**: Production Ready
