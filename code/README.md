# Stock Prediction Project - COS30018 Option C

## Project Overview
A comprehensive stock prediction system implementing and comparing multiple LSTM approaches for financial forecasting. This project demonstrates the evolution from basic tutorial code to advanced production-ready implementations with significant performance improvements.

### Key Highlights
- **28x Profit Improvement**: Advanced P1 model generates $915 vs basic v0.1's $32
- **45% Error Reduction**: P1 achieves 2.77 MAE vs v0.1's 5.07 MAE  
- **95.5% Accuracy**: P1 directional prediction accuracy vs v0.1's 53.7%
- **Multi-Feature Support**: Enhanced dev module supports Volume, OHLCV analysis
- **Production Ready**: Comprehensive data processing, caching, and evaluation metrics

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

### **v0.1 - Original YouTube Tutorial (Fixed)**
**Purpose**: Sequential split approach with next-day prediction
```bash
# Run the fixed v0.1 implementation
python v0.1/v0.1.py
```

**What it does:**
- Downloads AMZN stock data (last 2 years)
- Uses separate date ranges (distinct training and test periods)
- Predicts next-day prices
- Generates evaluation metrics and plots
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
- Uses configurable split (default: random 20% for testing)
- Predicts 15-day future prices
- Calculates trading-based accuracy metrics
- Outputs: `p1/results/p1_output.csv` and prediction plots

---

### 🔧 **dev - Current Development Module (adopted from v0.1 and P1)**
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

#### Multiple Features Prediction (In progress)
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
| **v0.1** | Separate date ranges | Next-day | 0.537 | 1 (Close) | ✅ Working |
| **p1** | Configurable (default: random) | 15-day | 0.955 | 5 (OHLCV) | ✅ Working |
| **dev** | Configurable | Next-day | ~0.52 | Configurable (OHLCV) | ✅ Working |

## Output Files
### v0.1 Outputs
- **Plot**: `v0.1/results/{COMPANY}_plot.png`

### p1 Outputs
- **Model**: `p1/results/*.weights.h5`
- **Plots**: `p1/results/{some_config}.png`
- **Trading CSV**: Multiple CSV files with buy/sell signals and profits

### dev Outputs
- **Model**: `dev/trained_models/*.h5`
- **Plots**: `dev/results/{some_config}.png`
- **Cache**: `dev/cache/processed_data/*.pkl` and `dev/cache/raw_data.pkl`

### All 3 use the same evaluating function for better comparison:
- **Full Evaluation**: `[folder]/results/info-eval.csv`

## Project Structure
```
stock-prediction-project/
├── v0.1/                   # Original YouTube tutorial code (fixed)
│   ├── v0.1.py             # Fixed stock prediction implementation
│   └── results/            # Output files and results
├── p1/                     # GitHub project implementation
│   ├── p1.py               # Core functions and data processing
│   ├── train.py            # Model training script
│   ├── test.py             # Model evaluation script
│   ├── parameters.py       # Configuration parameters
│   ├── p1.ipynb            # Jupyter notebook version
│   └── results/            # Model outputs and results
├── dev/                    # Advanced development module
│   ├── train.py            # Main training script
│   ├── training_data.py    # Cleaned data input for model training
│   ├── model/              # AI Model architect
│   ├── trained_models/     # Trained models cache
│   ├── data_preprocessing/ # Data processing modules
│   ├── cache/              # Cache processed data
│   ├── config/             # Configuration files
│   └── results/            # Output files (accuracy.csv & plots)
├── utils/                  # Shared utilities (data_handling, file_handling, eval, plots)
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

## Status
- ✅ **Task 1 Complete**: Environment setup, code testing, performance comparison
- ✅ **Task 2 Complete**: Data cleaning, code refactoring and modularization (for future extensibility & modifiablility :)
- 🚧 **Task 3 in progress**: Candlestick chart

---

**Last Updated**: August 22, 2025  
**Course**: COS30018 Option C