# Stock Prediction Project - COS30018 Option C

## Overview
This repository contains the implementation and analysis of stock prediction models for COS30018 Option C assignment. The project compares and improves upon different approaches to stock price prediction using LSTM neural networks.

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

### 🚀 **v0.1 - Original YouTube Tutorial (Fixed)**
**Purpose**: Sequential split approach with next-day prediction
```bash
# Run the fixed v0.1 implementation
python v0.1/v0.1.py
```

**What it does:**
- Downloads AMZN stock data (last 2 years)
- Uses sequential split (last 20% for testing)
- Predicts next-day prices
- Generates accuracy metrics and plots
- Outputs: `v0.1/results/AMZN_plot.png` and `v0.1/results/v0.1_output.csv`

---

### 🎯 **p1 - GitHub Project Implementation**
**Purpose**: Random split approach with 15-day prediction

#### Step 1: Train the Model
```bash
# Train the LSTM model
python p1/train.py
```

#### Step 2: Evaluate and Test
```bash
# Evaluate model performance and generate results
python p1/test.py
```

**What it does:**
- Downloads AMZN stock data (last 2 years)
- Uses random split (20% for testing)
- Predicts 15-day future prices
- Calculates trading-based accuracy metrics
- Outputs: `p1/results/p1_output.csv` and prediction plots

---

### 🔧 **dev - Advanced Development Module**
**Purpose**: Object-oriented approach with multiple features and models

#### Basic Usage (Default Settings)
```bash
# Use default settings (CBA.AX, Close price, 60 days)
python dev/train.py --model_name lstm
```

#### Single Feature Prediction
```bash
# Predict AAPL Close price with 30-day lookback
python dev/train.py --company AAPL --features Close --prediction_days 30 --model_name lstm
```

#### Multiple Features Prediction
```bash
# Use multiple features for better prediction
python dev/train.py --company AAPL --features Close Volume Open --prediction_days 60 --model_name lstm
```

#### Different Split Methods
```bash
# Date-based split (recommended for time series)
python dev/train.py --split_method date --test_size 0.2 --model_name lstm

# Random split (not recommended for time series)
python dev/train.py --split_method random --test_size 0.2 --shuffle --model_name lstm
```

#### Bidirectional LSTM Model
```bash
# Use bidirectional LSTM for potentially better performance
python dev/train.py --company AAPL --model_name bidirectional_lstm
```

#### Advanced Configuration
```bash
# Complete example with all features
python dev/train.py \
    --company AAPL \
    --features Close Volume Open High Low \
    --prediction_days 60 \
    --split_method date \
    --test_size 0.2 \
    --scale \
    --model_name lstm
```

**What it does:**
- Supports multiple companies and features
- Advanced data preprocessing with caching
- Multiple model architectures (LSTM, Bidirectional LSTM)
- Comprehensive evaluation metrics
- Outputs: Model files, plots, and accuracy CSV files

---

## Command Line Arguments (dev module)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--company` | str | CBA.AX | Company ticker symbol |
| `--start_date` | str | 2 years ago | Start date for data |
| `--end_date` | str | Today | End date for data |
| `--features` | list | ['Close'] | Features to use (Close, Volume, Open, High, Low, AdjClose) |
| `--prediction_days` | int | 60 | Number of days to look back |
| `--test_size` | float | 0.2 | Test set size ratio |
| `--split_method` | str | 'date' | Split method (date, ratio, random) |
| `--shuffle` | flag | True | Shuffle data (for random split) |
| `--scale` | flag | True | Scale features |
| `--model_name` | str | Required | Model type (lstm, bidirectional_lstm) |

## Performance Comparison

| Module | Split Method | Prediction | Accuracy Score | Features | Status |
|--------|-------------|------------|----------------|----------|---------|
| **v0.1** | Sequential (last 20%) | Next-day | ~0.45 | 1 (Close) | ✅ Working |
| **p1** | Random (20%) | 15-day | ~0.97 | 5 (OHLCV) | ✅ Working |
| **dev** | Configurable | Next-day | ~0.52 | Configurable | ✅ Working |

## Output Files

### v0.1 Outputs
- **Plot**: `v0.1/results/AMZN_plot.png`
- **Accuracy**: `v0.1/results/v0.1_output.csv`
- **Next Day**: `v0.1/results/AMZN_next_day_*.csv`

### p1 Outputs
- **Model**: `p1/results/*.weights.h5`
- **Accuracy**: `p1/results/p1_output.csv`
- **Plots**: Generated during test execution

### dev Outputs
- **Model**: `dev/trained_models/*.h5`
- **Plots**: `dev/results/{company}_{feature}_{days}_predictions.png`
- **Accuracy**: `dev/results/{company}_{feature}_{days}_accuracy.csv`
- **Cache**: `dev/cache/raw_data/*.pkl`

## Project Structure
```
stock-prediction-project/
├── v0.1/                    # Original YouTube tutorial code (fixed)
│   ├── v0.1.py             # Fixed stock prediction implementation
│   └── results/            # Output files and results
├── p1/                     # GitHub project implementation
│   ├── p1.py              # Core functions and data processing
│   ├── train.py           # Model training script
│   ├── test.py            # Model evaluation script
│   ├── parameters.py      # Configuration parameters
│   ├── p1.ipynb          # Jupyter notebook version
│   └── results/           # Model outputs and results
├── dev/                    # Advanced development module
│   ├── train.py           # Main training script
│   ├── model/             # Model implementations
│   ├── data_preprocessing/ # Data processing modules
│   ├── utils/             # Utility functions
│   ├── config/            # Configuration files
│   └── results/           # Output files
├── utils/                  # Shared utilities
│   └── accuracy_utils.py  # Shared accuracy calculation functions
├── requirements.txt        # Package dependencies
└── README.md              # This file
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

## Status
- ✅ **Task 1 Complete**: Environment setup, code testing, performance comparison
- ✅ **Task 2 Complete**: Code refactoring and modularization
- 🚧 **Task 3 in progress**: Advanced model improvements

---

**Last Updated**: August 22, 2025  
**Course**: COS30018 Option C