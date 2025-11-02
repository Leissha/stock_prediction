# Stock Prediction Project - COS30018 Option C

## Project Overview
This project demonstrates a comprehensive stock prediction system with sentiment analysis, multiple AI models, and classification capabilities for academic research.

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

### 2. Navigate to Development Directory
```bash
cd dev
```

### 3. Run Basic Examples
```bash
# Basic LSTM prediction
python main.py --company AAPL --use_sentiment

# Classification with sentiment analysis
python main.py --classification --use_sentiment --company AAPL

# Ensemble model
python main.py --model_name ensemble --ensemble_2 rnn --use_sentiment
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

#### Classification Mode
```bash
# Binary classification (up/down prediction)
python main.py --classification --use_sentiment --company AAPL

# Classification with different models
python main.py --classification --use_sentiment --company TSLA --scale
```

### Sentiment Integration

Sentiment from Google News RSS is integrated before the split and used as extra features.

Flags and behavior:

```bash
# Add FinBERT sentiment features (daily aggregation merged into stock df)
python main.py --use_sentiment

# Typical combined run (log returns + sentiment)
python main.py --company AAPL --log_ret --use_sentiment
```

Details:
- Source: Google News RSS (`company OR ticker` + date window)
- Caching: append-only CSV per ticker under `dev/cache/sentiment/{TICKER}_news.csv`
  - Only missing edge ranges are fetched; otherwise cached window is used
  - Deduplication by `url`, then `(title,date)`; dates normalized to daily
- Analyzer: FinBERT (`ProsusAI/finbert`) via `transformers/torch`
- Daily features: `sentiment_mean`, `news_count`
- Missing days: time-aware interpolation then fill 0
- Plot: unified train/val/test price vs smoothed sentiment saved to `results/{meta}_sentiment_vs_price_splits.png`

Note on instruments: for Australian tickers, include exchange suffix (e.g., `CBA.AX`).

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
- **Plots**: training curves `cache/trained_models/{meta}_training.png`, predictions `results/{meta}_predictions.png`, unified sentiment overlay `results/{meta}_sentiment_vs_price_splits.png`
- **Cache**: `cache/processed_data/*.pkl` and `cache/raw_data/*.pkl`
- **Results**: `results/{some_config}.csv`

## Project Structure
```
stock-prediction-project/
├── dev/                    # Main development module
│   ├── main.py             # CLI interface and orchestration
│   ├── pipeline.py         # Core data processing pipeline
│   ├── model/              # AI Model architectures
│   │   ├── tf_models.py    # TensorFlow models (LSTM, BiLSTM, GRU, RNN)
│   │   ├── sarimax.py      # SARIMAX univariate model
│   │   └── ensemble.py     # Weighted SARIMA+LSTM ensemble
│   ├── dataio/             # Pre/post-processing
│   │   ├── loading.py      # yfinance download + cache
│   │   ├── converters.py   # clean_data, build_target, time_split, scale_features, windows_*
│   │   └── postprocess.py  # descale, returns_to_prices, align_predictions
│   ├── eval/               # Evaluation
│   │   ├── regression_evaluator.py  # RegressionEvaluator class
│   │   ├── classification_evaluator.py  # ClassificationEvaluator class
│   │   └── metrics.py      # MAE, RMSE, DA
│   ├── sentiment/          # Sentiment Analysis
│   │   ├── crawl_news.py   # Google News RSS fetching
│   │   ├── sentiment_analyzer.py  # FinBERT sentiment analysis
│   │   └── sentiment_cache.py  # SentimentCache class
│   ├── schemas/            # Data contract
│   │   └── bundle.py       # DataBundle (Pydantic)
│   ├── config/             # Configuration files
│   │   ├── data.py         # Central defaults & date helpers
│   │   └── pipeline_config.py  # Pipeline configuration management
│   ├── utils/              # Utilities
│   │   ├── plots.py        # Plotting utilities
│   │   └── file_handling.py  # File operations
│   ├── cache/              # Cache processed data & models
│   │   ├── raw_data/       # Cached raw stock data
│   │   ├── processed_data/ # Cached processed sequences
│   │   ├── scalers/        # Cached feature scalers
│   │   ├── sentiment/      # Cached news and sentiment data
│   │   └── trained_models/ # Saved model files
│   ├── results/            # Output files (CSV & plots)
│   ├── streamlit/          # Optional web dashboard
│   ├── backend/            # Optional API backend
│   └── requirements.txt    # Package dependencies
├── requirements.txt        # Root package dependencies
└── README.md               # This file
```

## Architecture & Data Contracts

### Core Components
- **`DataBundle`**: Pydantic model enforcing shapes and metadata:
  - `X_train/X_val/X_test`: `(N, L, F)`, `y_*`: `(N, K)`
  - `lookback=L`, `horizon=K`, `target_mode` in {price, return, log_return}
  - Optional `train_df/test_df/val_df` for SARIMA, `base_prices_*` for price conversion

### Data Processing Pipeline
- **Preprocessing**: Single source of truth in `dataio.converters`:
  - `clean_data()`, `build_target()`, `time_split()`, `scale_features()`, `windows_train_test()`, `windows_train_val_test()`
- **Postprocessing**: In `dataio.postprocess`:
  - `descale()`, `returns_to_prices()`, `align_predictions()`; `predictions_to_prices()` used in pipeline

### Model Architecture
- **Models**: Implement only `fit()` and `predict()`; no internal scaling or metrics
- **Ensemble**: Weighted average after strict shape validation
- **Evaluation**: Centralized evaluators for regression and classification

### Sentiment Integration
- **`SentimentCache`**: Simplified sentiment analysis and caching
- **Features**: `sentiment_mean`, `news_count` (daily aggregation)
- **Caching**: Intelligent append-only CSV caching per ticker
- **Integration**: Pre-split sentiment feature merging

## Requirements
| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **Core ML** | tensorflow | ≥2.19.0 | Deep learning framework |
| | keras | ≥3.11.1 | High-level neural network API |
| | scikit-learn | ≥1.7.1 | Machine learning utilities |
| | numpy | ≥2.1.3 | Numerical computing |
| | pandas | ≥2.3.1 | Data manipulation |
| **Sentiment** | transformers | ≥4.40.0 | FinBERT sentiment analysis |
| | torch | ≥2.0.0 | PyTorch backend |
| **Visualization** | matplotlib | ≥3.10.5 | Plotting library |
| | seaborn | ≥0.13.0 | Statistical plotting |
| **Data Fetching** | yfinance | ≥0.2.65 | Yahoo Finance data |
| | requests | ≥2.32.4 | HTTP library |
| **Utilities** | loguru | ≥0.7.0 | Logging |
| | pydantic | ≥2.0.0 | Data validation |
| **Optional** | streamlit | ≥1.28.0 | Web dashboard |
| | fastapi | ≥0.100.0 | API backend |
| | uvicorn | ≥0.23.0 | ASGI server |

## Troubleshooting

### Common Issues
1. **Import Errors**: Make sure virtual environment is activated
2. **Data Download Issues**: Check internet connection and yfinance availability
3. **Ticker Suffixes**: Some exchanges require suffixes (e.g., `CBA.AX`). Using `CBA` will fetch a different US-listed instrument with small price levels.
4. **News Cache Messages**: "Fetching news..." logs indicate the pipeline step; verbose crawler logs will state whether cache was used or new articles were appended. Cache lives in `dev/cache/sentiment/`.
5. **Returns vs Prices in Plots**: If a plot shows decimal “prices”, it means a return series was picked. We now guard against this; ensure your DataFrame carries the `close` price column (log returns are stored in `close_log_return`).
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
- ✅ **Task 6 Complete**: Ensemble model implementation
- ✅ **Task 7 Complete**: Sentiment analysis integration with FinBERT

### Current Features
- **5 Execution Modes**: Sentiment, TF Models, SARIMAX, Classification, Ensemble
- **Sentiment Analysis**: FinBERT integration with intelligent caching
- **13 Model Variants**:
  - Base RNN: LSTM, GRU, RNN, BiLSTM
  - CNN Hybrid: CNN-LSTM, CNN-GRU, CNN-RNN, CNN-BiLSTM
  - Attention: Attention-LSTM, Attention-GRU, Attention-RNN, Attention-BiLSTM
  - Statistical: SARIMAX
  - Ensemble: SARIMAX + any TF model
- **Classification**: Binary up/down prediction with ablation study (with/without sentiment)
- **Multiple ML Models**: Logistic Regression, Random Forest, SVM for classification
- **Comprehensive Evaluation**: Regression and classification evaluators with ablation studies
- **Visualization**: Prediction plots, sentiment overlays, confusion matrices, training curves
- **Caching**: Multi-level caching (raw data, processed data, scalers, models, sentiment)

## Complete Model Catalog

### 1. Deep Learning Models (via TFModel)
All TensorFlow models support:
- Multi-step forecasting (`--horizon N`)
- Binary classification (`--classification`)
- Configurable architecture (`--layers 64 32 16`)
- Dropout regularization (`--dropout_rate 0.2`)
- Multiple optimizers (`--optimizer adam|rmsprop|sgd`)
- Early stopping with validation split (`--val_size 0.1`)

#### Base RNN Models
| Model | Command | Best For | Training Speed |
|-------|---------|----------|----------------|
| LSTM | `--model_name lstm` | General purpose, long-term dependencies | Medium |
| BiLSTM | `--model_name bilstm` | Full sequence context, bidirectional patterns | Slow |
| GRU | `--model_name gru` | Faster alternative to LSTM | Fast |
| RNN | `--model_name rnn` | Simple patterns, baseline | Very Fast |

#### CNN Hybrid Models
Combine Conv1D layers for spatial feature extraction with RNN for temporal modeling.

| Model | Command | Best For |
|-------|---------|----------|
| CNN-LSTM | `--model_name cnn_lstm` | Local patterns + long-term memory |
| CNN-GRU | `--model_name cnn_gru` | Local patterns + faster training |
| CNN-RNN | `--model_name cnn_rnn` | Local patterns + simple baseline |
| CNN-BiLSTM | `--model_name cnn_bilstm` | Local patterns + bidirectional context |

#### Attention Models
Use multi-head self-attention to focus on important time steps (e.g., earnings announcements).

| Model | Command | Best For |
|-------|---------|----------|
| Attention-LSTM | `--model_name attention_lstm` | Focus on key events + long memory |
| Attention-GRU | `--model_name attention_gru` | Focus on key events + faster |
| Attention-RNN | `--model_name attention_rnn` | Focus on key events + simple |
| Attention-BiLSTM | `--model_name attention_bilstm` | Focus on key events + bidirectional |

**Attention Configuration:**
- `--attn_heads 4`: Number of attention heads (default: 4)
- `--attn_key_dim 16`: Key dimension per head (default: 16)

### 2. Statistical Models

#### SARIMAX
Traditional time series model with seasonal components.

```bash
# Basic SARIMAX
python main.py --model_name sarimax --lookback 60 --horizon 5

# With seasonal components
python main.py --model_name sarimax --sarimax_seasonal --sarimax_m 5
```

**Parameters:**
- `--sarimax_seasonal`: Enable seasonal components
- `--sarimax_m 5`: Seasonal period (e.g., 5 for weekly patterns)

### 3. Ensemble Models

Weighted combination of SARIMAX (statistical) and any TF model (neural).

```bash
# SARIMAX + GRU ensemble (equal weights)
python main.py --model_name ensemble --ensemble_2 gru \
  --sarima_weight 0.5 --model_2_weight 0.5

# SARIMAX + BiLSTM ensemble (favor neural)
python main.py --model_name ensemble --ensemble_2 bilstm \
  --sarima_weight 0.3 --model_2_weight 0.7
```

**Parameters:**
- `--ensemble_2 <lstm|gru|rnn|bilstm>`: Choose TF model for ensemble
- `--sarima_weight 0.5`: Weight for SARIMAX predictions
- `--model_2_weight 0.5`: Weight for TF model predictions

### 4. Classification Models

Binary up/down prediction with multiple models and ablation study.

```bash
# LSTM classification with sentiment analysis
python main.py --classification --use_sentiment --model_name lstm

# Compare multiple classifiers
python main.py --classification --use_sentiment
```

**Models Trained:**
1. Time Series: LSTM (or specified model)
2. Logistic Regression (with class balancing)
3. Random Forest (with class balancing)
4. SVM (with class balancing)

**Automatic Ablation Study:**
- Trains baseline model without sentiment
- Compares performance with/without sentiment features
- Generates impact analysis plots

**Classification Approaches:**
- Default: Direct binary classification with BCE loss + optimal threshold tuning
- Alternative: `--use_price_comparison` trains regression, converts to binary via price comparison

## Model Selection Guide

### For Best Accuracy (regardless of compute)
1. **BiLSTM** or **Attention-BiLSTM** with sentiment
2. **CNN-BiLSTM** if local patterns matter
3. **Ensemble** (SARIMAX + BiLSTM) for robustness

### For Fastest Training
1. **GRU** - fastest neural model
2. **RNN** - simplest baseline
3. **SARIMAX** - no neural training needed

### For Interpretability
1. **SARIMAX** - traditional statistical model with interpretable coefficients
2. **Attention models** - visualize which time steps matter most

### For Resource-Constrained Environments
1. **GRU** or **RNN**
2. Use smaller `--layers 32 16` architecture
3. Reduce `--epochs 25` for faster training

### For Research/Experimentation
1. **Attention models** - investigate what the model learns
2. **Ensemble** - compare statistical vs neural approaches
3. **Classification with ablation** - measure sentiment impact quantitatively

---

**Last Updated**: November 2, 2025
**Course**: COS30018 Option C
**Total Models**: 13 variants (4 base RNN + 4 CNN hybrid + 4 attention + 1 statistical + ensemble)
