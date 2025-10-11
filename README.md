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
# Predict different target features (Close, Open, High, Low, Volume)
python main.py --company AAPL --target_feature Close --lookback 60 --horizon 5 --model_name lstm --scale
python main.py --company TSLA --target_feature Open  --lookback 90 --horizon 1 --model_name gru  --scale
```


#### Validation Split
```bash
# Enable validation split passed to Keras (EarlyStopping auto-enabled)
python main.py --model_name lstm --lookback 60 --horizon 5 --scale --val_size 0.1
```

#### Model Selection
```bash
# Standard LSTM (default)
python main.py --model_name lstm

# Bidirectional LSTM for potentially better performance
python main.py --model_name bilstm

# GRU / Simple RNN
python main.py --model_name gru
python main.py --model_name rnn
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
# Complete example
python main.py \
    --company AAPL \
    --start_date 2022-01-01 \
    --end_date 2024-01-01 \
    --target_feature Close \
    --lookback 60 \
    --horizon 5 \
    --scale \
    --model_name lstm \
    --layers 64 32 \
    --dropout_rate 0.2 \
    --epochs 50 \
    --batch_size 32 \
    --optimizer adam \
    --learning_rate 0.001 \
    --val_size 0.1 \
    --inspect_plots
```

**What it does (v2 pipeline):**
- **Unified shapes**: `X (N,L,F)`, `y (N,K)`, `preds (N,K)`
- **Single source of truth**: clean/target/split/scale/window in `dev/dataio/converters.py`, orchestrated by `dev/pipeline.py`
- **Models**: `fit(...)`, `predict(X)->(N,K)` only; no internal scaling or metrics
- **Validation split**: `--val_size` routes to Keras `validation_data` with EarlyStopping
- **Postprocess & metrics**: `dev/dataio/postprocess.py`, `dev/eval/metrics.py`
- **Caching**: raw data, models, results under `dev/cache` and `dev/results`

#### SARIMA & Ensemble
```bash
# SARIMA (univariate target)
python main.py --model_name sarimax --lookback 60 --horizon 5 --sarimax_seasonal --sarimax_m 5

# Ensemble: SARIMA + GRU (weighted average)
python main.py --model_name ensemble --ensemble_2 gru --lookback 60 --horizon 5 \
  --layers 64 32 --dropout_rate 0.2 --epochs 50 --batch_size 32 \
  --optimizer rmsprop --learning_rate 0.0005 \
  --sarima_weight 0.5 --model_2_weight 0.5 \
  --scale --val_size 0.1 --log_ret
```

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
| `--lookback` | int | 60 | Number of days to look back for prediction |
| `--horizon` | int | 1 | **Number of future days to predict (1=single-step, >1=multistep)** |
| `--test_size` | float | 0.2 | Test set size ratio (chronological) |
| `--scale` | flag | True | Scale features |
| `--model_name` | str | lstm | Model type (lstm, gru, rnn, bilstm, sarimax, ensemble) |
| `--val_size` | float | 0.2 | Validation set size ratio |
| `--sarimax_seasonal` | flag | False | Enable seasonal SARIMAX |
| `--sarimax_m` | int | 5 | Seasonal period |
| `--sarima_weight` | float | 0.2 | Ensemble SARIMA weight |
| `--model_2_weight` | float | 0.8 | Ensemble TF-model weight |
| `--ensemble_2` | str | lstm | TF submodel used by ensemble (lstm, gru, rnn, bilstm) |
| `--layers` | list | [50, 50, 50] | Layer sizes (e.g., 64 32 16) |
| `--dropout_rate` | float | 0.2 | Dropout rate for regularization |
| `--epochs` | int | 50 | Number of training epochs |
| `--batch_size` | int | 32 | Batch size for training |
| `--optimizer` | str | adam | Optimizer (adam, rmsprop, sgd) |
| `--learning_rate` | float | 0.001 | Learning rate for optimizer |
| `--inspect_plots` | flag | False | Generate preprocessing plots (candlestick/boxplot) |

## Output Files

- **Model**: `cache/trained_models/*.keras`
- **Plots**: training curves `cache/trained_models/{meta}_training.png`, predictions `results/{meta}_predictions.png`
- **Cache**: `cache/processed_data/*.pkl` and `cache/raw_data/*.pkl`
- **Results**: `results/{some_config}.csv`

## Project Structure
```
stock-prediction-project/
├── dev/                    # Advanced development module
│   ├── main.py             # CLI interface and orchestration
│   ├── model/              # AI Model architectures
│   │   ├── tf_models.py    # TensorFlow models (LSTM, BiLSTM, GRU, RNN)
│   │   ├── sarimax.py      # SARIMAX univariate model
│   │   └── ensemble.py     # Weighted SARIMA+LSTM ensemble
│   ├── dataio/             # Pre/post-processing
│   │   ├── loading.py      # yfinance download + cache
│   │   ├── converters.py   # clean_data, build_target, time_split, scale_features, windows_*
│   │   └── postprocess.py  # descale, returns_to_prices, align_predictions
│   ├── eval/               # Evaluation
│   │   └── metrics.py      # MAE, RMSE, DA
│   ├── schemas/            # Data contract
│   │   └── bundle.py       # DataBundle (Pydantic)
│   ├── config/             # Configuration files
│   │   └── data.py         # Central defaults & date helpers
│   ├── cache/              # Cache processed data & models
│   │   ├── raw_data/       # Cached raw stock data
│   │   ├── processed_data/ # Cached processed sequences
│   │   ├── scalers/        # Cached feature scalers
│   │   └── trained_models/ # Saved model files
│   └── results/            # Output files (CSV & plots)
├── requirements.txt        # Package dependencies
└── README.md               # This file
```

## Architecture & Data Contracts

- `DataBundle` enforces shapes and metadata:
  - `X_train/X_val/X_test`: `(N, L, F)`, `y_*`: `(N, K)`
  - `lookback=L`, `horizon=K`, `target_mode` in {price, return, log_return}
  - Optional `train_df/test_df/val_df` for SARIMA, `base_prices_*` for price conversion
- Preprocessing single source of truth in `dataio.converters`:
  - `clean_data()`, `build_target()`, `time_split()`, `scale_features()`, `windows_train_test()`, `windows_train_val_test()`
- Postprocessing in `dataio.postprocess`:
  - `descale()`, `returns_to_prices()`, `align_predictions()`; `predictions_to_prices()` used in pipeline
- Models implement only `fit()` and `predict()`; no internal scaling or metrics
- Ensemble does weighted average after strict shape validation

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

**Last Updated**: 11 October, 2025  
**Course**: COS30018 Option C
