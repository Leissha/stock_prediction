## Glossary (symbols used throughout the code)

### Shape Notation
- **X**: model inputs of shape **(N, L, F)**
  - **N**: number of samples (windows)
  - **L**: lookback length (sequence length) — aliased as `lookback` in code
  - **F**: number of features (automatically inferred from data shape)
- **y**: targets of shape **(N, K)**
  - **K**: forecast horizon (steps ahead) — aliased as `horizon` or `forecast_horizon` in code
- **y_hat**: model predictions of shape **(N, K)**

### Data Modes
- **mode**: target type; one of `"price"` | `"return"` | `"log_return"`
  - `price`: raw price values (no transformation)
  - `return`: simple returns = (p_t - p_{t-1}) / p_{t-1}
  - `log_return`: log returns = log(p_t / p_{t-1})

### Key Variables
- **base_prices_test**: per-sample base price at time t for test rows (shape: N,)
  - Used to convert returns → prices for evaluation
  - Critical for multi-step forecasting
- **DataBundle**: validated Pydantic model carrying all arrays and metadata
  - Enforces shape contracts: X:(N,L,F), y:(N,K)
  - Contains: `lookback=L`, `horizon=K`, `target_mode`, `scalers`

### Loop Variables (Standard Conventions)
- **sample_idx**: index over samples (0 to N-1)
- **step**: index over forecast horizon (0 to K-1)
- **current_price**: running price during return compounding

## Naming Conventions

### Model Interface
- All model classes must implement: `predict(X_test, y_test=None) -> np.ndarray`
  - Returns shape: **(N, K)**
  - `y_test` is optional (for models that need it during prediction)
- Store all training history/artifacts in model attributes for later access

---

## Component Architecture

### High-Level Overview
```mermaid
graph TB
    CLI[CLI / main.py] --> PIPELINE[pipeline.py]

    subgraph DATAIO[dataio/ - Data I/O]
        LOAD[loading.py<br/>load_stock_data]
        CONV[converters.py<br/>clean_data, build_target, time_split, scale_features, windows_train_test, windows_train_val_test]
        POST[postprocess.py<br/>descale, returns_to_prices]
    end

    subgraph SCHEMAS[schemas/ - Data Contract]
        BUNDLE[bundle.py<br/>DataBundle Pydantic Model]
    end

    subgraph MODELS[model/ - ML Models]
        TF[tf_models.py<br/>LSTM/GRU/RNN/BiLSTM]
        SARIMA[sarimax.py<br/>SARIMAX]
        ENS[ensemble.py<br/>Weighted Average]
    end

    subgraph SENTIMENT[sentiment/ - Sentiment Analysis]
        CRAWL[crawl_news.py<br/>Google News RSS]
        ANALYZER[sentiment_analyzer.py<br/>FinBERT]
        CACHE[sentiment_cache.py<br/>SentimentCache]
    end

    subgraph EVAL[eval/ - Evaluation]
        REGRESSION[regression_evaluator.py<br/>RegressionEvaluator]
        CLASSIFICATION[classification_evaluator.py<br/>ClassificationEvaluator]
        METRICS[metrics.py<br/>MAE, RMSE, DA]
    end

    subgraph UTILS[utils/ - Utilities]
        PLOTS[plots.py]
        FILES[file_handling.py]
    end

    PIPELINE --> LOAD
    LOAD --> CONV
    CONV --> BUNDLE
    BUNDLE --> MODELS
    MODELS --> POST
    POST --> EVAL
    EVAL --> UTILS

    style DATAIO fill:#e1f5ff
    style SCHEMAS fill:#fff4e1
    style MODELS fill:#f0e1ff
    style EVAL fill:#e1ffe1
```

---

## Data Flow Pipeline

### Detailed Processing Steps

```mermaid
graph LR
    subgraph Input
        RAW[Raw CSV/Yahoo<br/>OHLCV Data]
    end

    subgraph Stage1[Stage 1: Load & Clean]
        LOAD[load_stock_data<br/>cache/raw_data/*.pkl]
        CLEAN[clean_data<br/>ffill NaNs]
    end

    subgraph Stage2[Stage 2: Transform]
        TARGET[build_target<br/>price/return/log_return]
        SPLIT[time_split<br/>train/test chronological]
    end

    subgraph Stage2b[Stage 2.5: Sentiment]
        NEWS[get_stock_news<br/>Google News RSS + cache]
        FINBERT[FinBERT analyze<br/>ProsusAI/finbert]
        AGG[aggregate daily<br/>sentiment_mean/news_count]
        MERGE[merge into df<br/>pre-split]
    end

    subgraph Stage3[Stage 3: Scale]
        SCALE[scale_features<br/>StandardScaler per-feature]
    end

    subgraph Stage4[Stage 4: Window]
        WINDOW[windows_train_val_test<br/>X:N,L,F and y:N,K]
    end

    subgraph Stage5[Stage 5: Validate]
        BUNDLE[DataBundle<br/>Pydantic validation]
    end

    subgraph Output
        TRAIN[X_train, y_train]
        VAL[X_val, y_val]
        TEST[X_test, y_test]
    end

    RAW --> LOAD
    LOAD --> CLEAN
    CLEAN --> TARGET
    TARGET --> NEWS
    NEWS --> FINBERT
    FINBERT --> AGG
    AGG --> MERGE
    MERGE --> SPLIT
    TARGET --> SPLIT
    SPLIT --> SCALE
    SCALE --> WINDOW
    WINDOW --> BUNDLE
    BUNDLE --> TRAIN
    BUNDLE --> VAL
    BUNDLE --> TEST
```

---

## Function Reference

### dataio/converters.py

#### `clean_data(df, method="ffill") -> DataFrame`
- **Purpose**: Handle missing values
- **Input**: Raw DataFrame with OHLCV columns
- **Output**: Cleaned DataFrame (M', C) where M' <= M
- **Methods**: `"ffill"` (forward-fill) or `"drop"`

#### `build_target(df, target_col, mode, use_log=False) -> (Series, str)`
- **Purpose**: Create target variable based on mode
- **Input**: DataFrame, target column name, mode
- **Output**: (target_series, target_column_name)
- **Modes**:
  - `"price"`: return original prices
  - `"return"`: compute pct_change() or log returns if use_log=True
  - `"log_return"`: compute log(p_t / p_{t-1})
- **Note**: Returns drop first row (NaN from diff) for return modes

#### `time_split(df, test_size=0.2, val_size=0.0) -> (train_df, test_df, val_df)`
- **Purpose**: Chronological train/val/test split
- **Input**: 
  - `df`: Time-indexed DataFrame
  - `test_size`: Proportion for test set (0.0-1.0)
  - `val_size`: Proportion for validation set (0.0-1.0)
- **Output**: (train_df, test_df, val_df) - val_df is None if val_size=0
- **Note**: NO shuffling - preserves temporal order

#### `scale_features(train_df, test_df, feature_cols) -> (train_scaled, test_scaled, scalers)`
- **Purpose**: Fit StandardScaler on train, transform both
- **Input**: Train/test DataFrames, feature column names
- **Output**: (train_scaled: (T,F), test_scaled: (Te,F), scalers: dict)
- **Note**: Separate scaler per feature to prevent leakage

#### `windows_train_test(features_scaled, target_series, lookback, horizon=1) -> (X, y)`
- **Purpose**: Create sliding windows for time series prediction (primitive function)
- **Input**:
  - `features_scaled`: (M, F) array
  - `target_series`: (M,) array - single target column
  - `lookback`: L - sequence length
  - `horizon`: K - forecast steps (default 1)
- **Output**:
  - `X`: (N, L, F) - N sliding windows
  - `y`: (N, K) - future targets
- **Formula**: N = M - L - K
- **Note**: F is inferred from `features_scaled.shape[1]`

#### `windows_train_val_test(train_scaled, test_scaled, target_train, target_test, lookback, horizon, val_scaled=None, target_val=None) -> (X_train, y_train, X_test, y_test, X_val, y_val)`
- **Purpose**: Orchestrate windowing for train/val/test splits with primitive alignment
- **Input**:
  - `train_scaled`, `test_scaled`: (M, F) arrays
  - `target_train`, `target_test`: (M,) arrays
  - `lookback`: L - sequence length
  - `horizon`: K - forecast steps
  - `val_scaled`, `target_val`: Optional validation arrays
- **Output**: All windowed arrays with consistent alignment
- **Note**: Single entrypoint for creating all data windows

**Example:**
```python
# Create 60-day sequences to predict next 5 days
features = np.random.rand(500, 5)  # 500 days, 5 features
target = np.random.rand(500)        # 500 target values

X, y = windows_train_test(features, target, lookback=60, horizon=5)
# X.shape = (435, 60, 5)  where 435 = 500 - 60 - 5
# y.shape = (435, 5)
```

---

### dataio/postprocess.py

#### `descale(arr, scaler) -> ndarray`
- **Purpose**: Inverse StandardScaler transformation
- **Input**:
  - `arr`: (N, K) scaled array
  - `scaler`: StandardScaler or None
- **Output**: (N, K) unscaled array
- **Note**: Handles 1D → 2D conversion automatically

#### `returns_to_prices(returns, base_prices, mode, use_log=False, validate=True) -> ndarray`
- **Purpose**: Convert returns → prices using compounding
- **Input**:
  - `returns`: (N, K) return predictions
  - `base_prices`: (N,) base price at time t for each sample
  - `mode`: "price" | "return" | "log_return"
  - `use_log`: whether to use log returns
- **Output**: (N, K) prices
- **Formulas**:
  - Simple return: `price[k] = base * prod(1 + r[0:k+1])`
  - Log return: `price[k] = base * prod(exp(r[0:k+1]))`
- **Variables**:
  - `sample_idx`: iterates over N samples
  - `step`: iterates over K forecast steps
  - `current_price`: running compounded price

**Example:**
```python
returns = np.array([[0.02, 0.01]])  # 2%, 1% returns
base = np.array([100.0])             # base price = 100

prices = returns_to_prices(returns, base, mode="return")
# prices[0, 0] = 100 * (1 + 0.02) = 102.0
# prices[0, 1] = 102 * (1 + 0.01) = 103.02
```

#### `prices_to_returns(prices, base_prices, mode="return", use_log=False) -> ndarray`
- **Purpose**: Inverse of returns_to_prices() - convert prices → returns
- **Input**:
  - `prices`: (N, K) price predictions
  - `base_prices`: (N,) base prices
- **Output**: (N, K) returns
- **Note**: Computes sequential returns: r[k] = (price[k] - price[k-1]) / price[k-1]

---

### pipeline.py

#### `predictions_to_prices(y_true, y_hat, bundle) -> (y_true_prices, y_hat_prices)`
- **Purpose**: Convert predictions to price space for evaluation (pipeline-level function)
- **Input**:
  - `y_true`: (N, K) ground truth predictions
  - `y_hat`: (N, K) model predictions  
  - `bundle`: DataBundle with metadata and scalers
- **Output**: (y_true_prices, y_hat_prices) both (N, K)
- **Process**: 
  1. Descale both arrays using `bundle.target_scaler`
  2. Convert to prices using `returns_to_prices()` with `bundle.base_prices_test`
- **Note**: Used in pipeline for final evaluation in price space

---

### eval/regression_evaluator.py

#### `RegressionEvaluator`
- **Purpose**: Centralized regression model evaluation
- **Input**: DataBundle, model predictions, configuration
- **Output**: Metrics, plots, CSV results
- **Features**:
  - Price space conversion
  - Comprehensive metrics (MAE, RMSE, DA)
  - Prediction plots
  - Sentiment visualization (if enabled)

#### `run_regression_evaluation(args, bundle, model, meta_path)`
- **Purpose**: Single entry point for regression evaluation
- **Process**: 
  1. Generate predictions
  2. Convert to price space
  3. Calculate metrics
  4. Create plots
  5. Save results

---

### eval/classification_evaluator.py

#### `ClassificationEvaluator`
- **Purpose**: Binary classification evaluation with ablation study
- **Input**: Ticker, date range, sentiment flag, configuration
- **Output**: Classification metrics, confusion matrices, ablation results
- **Features**:
  - Multiple models (LSTM, Logistic Regression, Random Forest, SVM)
  - Baseline comparison (with/without sentiment)
  - Comprehensive metrics (accuracy, precision, recall, F1)
  - Visualization plots

#### `run_classification_evaluation(ticker, start_date, end_date, use_sentiment, scale, test_size, val_size)`
- **Purpose**: Single entry point for classification evaluation
- **Process**:
  1. Prepare data using main pipeline
  2. Train multiple classification models
  3. Run ablation study
  4. Generate plots and save results

---

### sentiment/sentiment_cache.py

#### `SentimentCache`
- **Purpose**: Simplified sentiment analysis and caching
- **Features**:
  - News fetching with Google RSS
  - FinBERT sentiment analysis
  - Intelligent caching (append-only)
  - Daily sentiment aggregation
  - Integration with stock data

#### `get_news_with_sentiment(ticker, start_date, end_date) -> DataFrame`
- **Purpose**: Get news with sentiment analysis
- **Process**:
  1. Check existing cache
  2. Fetch missing articles
  3. Analyze sentiment for new articles
  4. Cache results
- **Output**: DataFrame with news and sentiment scores

#### `integrate_sentiment(df, ticker, start_date, end_date) -> DataFrame`
- **Purpose**: Integrate sentiment features into stock data
- **Process**:
  1. Get daily sentiment aggregation
  2. Merge with stock data
  3. Forward-fill missing values
- **Output**: DataFrame with sentiment features added

---

### dataio/loading.py

#### `load_stock_data(company, start_date, end_date, cache_dir='cache/raw_data') -> DataFrame`
- **Purpose**: Download/cache stock data from Yahoo Finance
- **Input**: Ticker symbol, date range, cache directory
- **Output**: DataFrame with OHLCV columns
- **Caching**: Saves to `{cache_dir}/{ticker}_{start}_to_{end}.pkl`
- **Note**: Flattens MultiIndex columns from yfinance

---

## Model Interface Contract

All models in `dev/model/` must follow this interface:

```python
class ModelInterface:
    """Abstract model interface (not enforced, but conventional)"""

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Train the model.

        Args:
            X_train: (N, L, F) training sequences
            y_train: (N, K) training targets
            **kwargs: Model-specific parameters
        """
        pass

    def predict(self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X_test: (N_test, L, F) test sequences
            y_test: (N_test, K) optional ground truth (for models that need it)

        Returns:
            y_hat: (N_test, K) predictions
        """
        pass
```

### Model Implementations

#### TFModel (LSTM/GRU/RNN/BiLSTM)
```python
# Located in: dev/model/tf_models.py
model = TFModel(config)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)  # Returns (N, K)
```

#### SARIMAXModel
```python
# Located in: dev/model/sarimax.py
model = SARIMAXModel(forecast_horizon=K)
model.fit(train_df, target_feature="close_return", scalers=scalers)
y_hat = model.predict(X_test)  # Returns (N, K)
```

#### EnsembleModel
```python
# Located in: dev/model/ensemble.py
config = EnsembleConfig(sarima_weight=0.3, model_2_weight=0.7)
ensemble = EnsembleModel(config)
ensemble.fit(sarima_model, lstm_model)  # Pre-trained models
y_hat = ensemble.predict(X_test)  # Weighted average (N, K)
```

---

## DataBundle Schema

The `DataBundle` Pydantic model enforces strict shape contracts:

```python
@dataclass
class DataBundle:
    # Core data
    X_train: NDArray  # (N_train, L, F)
    y_train: NDArray  # (N_train, K)
    X_test: NDArray   # (N_test, L, F)
    y_test: NDArray   # (N_test, K)

    # Metadata
    symbol: str
    features: List[str]  # Length F
    lookback: int        # L
    horizon: int         # K
    target_mode: TargetMode  # PRICE | RETURN | LOG_RETURN
    use_log_returns: bool

    # Scalers
    scalers: Dict[str, StandardScaler]
    target_scaler: Optional[StandardScaler]
    base_prices_test: Optional[NDArray]  # (N_test,)

    # SARIMAX support
    train_df: Optional[DataFrame]
    test_df: Optional[DataFrame]
    target_feature: Optional[str]
```

**Validations enforced:**
- X.ndim == 3, y.ndim == 2
- X.shape[0] == y.shape[0] (sample count match)
- X.shape[1] == lookback
- X.shape[2] == len(features)
- y.shape[1] == horizon
- base_prices_test.shape[0] == X_test.shape[0]
- target_mode consistency with use_log_returns

---

## End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant CLI as main.py
    participant PL as pipeline.py
    participant LD as loading.load_stock_data
    participant CV as converters
    participant SN as sentiment_cache
    participant SC as DataBundle
    participant M as Model
    participant PP as postprocess

    CLI->>PL: prepare_data(ticker, dates, params)

    Note over PL,LD: Stage 1: Load & Clean
    PL->>LD: load_stock_data(ticker, start, end)
    LD-->>PL: DataFrame (M rows, OHLCV)
    PL->>CV: clean_data(df)
    CV-->>PL: df_clean

    Note over PL,CV: Stage 2: Transform Target
    PL->>CV: build_target(df, col, mode, use_log)
    CV-->>PL: (target_series, target_name)
    PL->>PL: align df to target.index

    Note over PL,SN: Stage 2.5: Sentiment (optional)
    PL->>SN: integrate_sentiment(df, ticker, start, end)
    SN-->>PL: df_with_daily_sentiment

    Note over PL,CV: Stage 3: Split & Scale
    PL->>CV: time_split(df, test_size=0.2)
    CV-->>PL: (train_df, test_df)
    PL->>CV: scale_features(train, test, features, val_df)
    CV-->>PL: (train_scaled, test_scaled, val_scaled, scalers)

    Note over PL,CV: Stage 4: Create Windows (with alignment)
    PL->>CV: windows_train_val_test(train_scaled, test_scaled, target_train, target_test, L, K, val_scaled)
    CV-->>PL: X_train:(N,L,F), y_train:(N,K)
    CV-->>PL: X_test:(N,L,F), y_test:(N,K)
    opt if val_scaled provided
        CV-->>PL: X_val:(N,L,F), y_val:(N,K)
    end

    Note over PL,SC: Stage 5: Validate Bundle
    PL->>SC: DataBundle(X, y, metadata)
    SC->>SC: Validate shapes, dims, consistency
    SC-->>PL: bundle (validated)

    Note over PL,M: Stage 6: Train & Predict
    PL->>M: fit(bundle.X_train, bundle.y_train)
    M-->>PL: trained model
    PL->>M: predict(bundle.X_test)
    M-->>PL: y_hat:(N,K)

    Note over PL,PP: Stage 7: Postprocess
    alt mode == return or log_return
        PL->>PP: descale(y_true, target_scaler)
        PP-->>PL: y_true_descaled
        PL->>PP: descale(y_hat, target_scaler)
        PP-->>PL: y_hat_descaled
        PL->>PP: returns_to_prices(y_true, base_prices, mode)
        PP-->>PL: y_true_prices:(N,K)
        PL->>PP: returns_to_prices(y_hat, base_prices, mode)
        PP-->>PL: y_hat_prices:(N,K)
    else mode == price
        PL->>PL: Use predictions as-is
    end

    Note over PL,CLI: Stage 8: Evaluate & Return
    PL->>PL: compute_metrics(y_true, y_hat)
    PL-->>CLI: (metrics, predictions, artifacts)
```

---

## Scaling Architecture

### Overview
The scaling system follows a **controlled, consistent approach** where the `--scale` flag determines whether both features AND targets are scaled together. This ensures no scale mismatches between inputs and outputs.

### Scaling Logic Matrix

| Mode | `--scale` Flag | Features Scaled? | Targets Scaled? | Result |
|------|----------------|------------------|-----------------|---------|
| **PRICE** | `--scale` |  Yes |  Yes |  **Consistent** |
| **PRICE** | No flag |  No |  No |  **Consistent** |
| **RETURN** | `--scale` |  Yes |  Yes |  **Consistent** |
| **RETURN** | No flag |  No |  No |  **Consistent** |
| **LOG_RETURN** | `--scale` |  Yes |  Yes |  **Consistent** |
| **LOG_RETURN** | No flag |  No |  No |  **Consistent** |

#### 1. Feature Scaling (`scale_features`)
```python
# Located in: dataio/converters.py
def scale_features(train_df, test_df, feature_cols, val_df=None):
    """
    Fit StandardScaler on training data, transform all splits.
    
    Process:
    1. Create separate StandardScaler for each feature
    2. Fit scalers ONLY on training data (prevents leakage)
    3. Transform train/test/val using fitted scalers
    4. Return scaled arrays and scaler dictionary
    """
    scalers = {}
    for i, col in enumerate(feature_cols):
        scaler = StandardScaler()
        # Fit on train only
        train_scaled[:, i] = scaler.fit_transform(train_df[[col]].values)
        # Transform test/val with same scaler
        test_scaled[:, i] = scaler.transform(test_df[[col]].values)
        if val_df is not None:
            val_scaled[:, i] = scaler.transform(val_df[[col]].values)
        scalers[col] = scaler
```

#### 2. Target Scaling (`scale_target`)
```python
# Located in: dataio/converters.py
def scale_target(target_train, target_test, target_val=None, scaler=None):
    """
    Scale target values using StandardScaler.
    
    Process:
    1. Create/fit StandardScaler on training targets only
    2. Transform all target splits using fitted scaler
    3. Return scaled targets and scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        # Fit on training data only
        target_train_scaled = scaler.fit_transform(target_train.reshape(-1, 1)).ravel()
    else:
        target_train_scaled = scaler.transform(target_train.reshape(-1, 1)).ravel()
    
    # Transform test/val with same scaler
    target_test_scaled = scaler.transform(target_test.reshape(-1, 1)).ravel()
    if target_val is not None:
        target_val_scaled = scaler.transform(target_val.reshape(-1, 1)).ravel()
```

#### 3. Pipeline Integration
```python
# Located in: pipeline.py
def prepare_data(..., scale=True, ...):
    # 1. Scale features (always when scale=True)
    if scale:
        train_scaled, test_scaled, val_scaled, scalers = scale_features(
            train_df, test_df, features, val_df=val_df
        )
    else:
        train_scaled = train_df[features].values
        test_scaled = test_df[features].values
        val_scaled = val_df[features].values if val_df is not None else None
        scalers = {}

    # 2. Scale targets (always when scale=True - ALL modes)
    target_scaler = None
    if scale:
        target_train, target_test, target_val, target_scaler = scale_target(
            target_train, target_test, target_val
        )
        scalers['__target__'] = target_scaler
```

### Scaling Flow Diagram

```mermaid
graph TB
    subgraph Input[Input Data]
        RAW[Raw OHLCV Data]
    end
    
    subgraph ScaleDecision[Scale Decision]
        SCALE[--scale flag]
        NOSCALE[No --scale flag]
    end
    
    subgraph ScaleProcess[Scaling Process]
        FEATSCALE[scale_features<br/>StandardScaler per feature<br/>Fit on train only]
        TARGETSCALE[scale_target<br/>StandardScaler for targets<br/>Fit on train only]
    end
    
    subgraph Train[Training]
        MODEL[Model Training<br/>scaled_features → scaled_targets]
    end
    
    subgraph Predict[Prediction]
        PRED[Model Prediction<br/>scaled_features → scaled_predictions]
    end
    
    subgraph Descale[Descale Process]
        DESCALETARGET[Descale Targets<br/>bundle.target_scaler.inverse_transform]
    end
    
    subgraph Output[Final Output]
        PRICES[Price Space Predictions<br/>Ready for evaluation]
    end
    
    RAW --> SCALE
    RAW --> NOSCALE
    SCALE --> FEATSCALE
    SCALE --> TARGETSCALE
    NOSCALE --> MODEL
    
    FEATSCALE --> MODEL
    TARGETSCALE --> MODEL
    MODEL --> PRED
    PRED --> DESCALETARGET
    DESCALETARGET --> PRICES
    
    style ScaleDecision fill:#fff2cc
    style ScaleProcess fill:#e1f5ff
    style Descale fill:#f0e1ff
    style Output fill:#e1ffe1
```

### Key Principles

#### 1. **Consistency**
- When `--scale` is used: **Both** features AND targets are scaled
- When `--scale` is NOT used: **Neither** features NOR targets are scaled
- No exceptions or mode-specific logic

#### 2. **No Data Leakage**
- Scalers are fitted **ONLY** on training data
- Test and validation data use the same fitted scalers
- No future information leaks into past scaling

#### 3. **Proper Descale**
- All predictions are descaled back to original space
- Descale happens **before** return-to-price conversion
- Maintains numerical precision throughout

#### 4. **Single Source of Truth**
- All scaling logic centralized in `dataio/converters.py`
- Pipeline orchestrates but doesn't implement scaling
- Consistent interface across all modes

### Example Scaling Values

#### PRICE Mode with Scaling
```python
# Original prices
train_prices = [100.0, 101.0, 102.0, 99.0, 103.0]

# After StandardScaler
scaled_prices = [-0.5, 0.0, 0.5, -1.0, 1.0]  # mean=0, std=1

# Model learns: scaled_features → scaled_prices
# Prediction: scaled_features → scaled_prediction

# Descale back to original space
descaled_prediction = 101.5  # Back to price units
```

#### RETURN Mode with Scaling
```python
# Original returns
train_returns = [0.01, 0.02, -0.01, 0.03, -0.02]

# After StandardScaler
scaled_returns = [-0.2, 0.4, -0.6, 0.8, -1.0]  # mean=0, std=1

# Model learns: scaled_features → scaled_returns
# Prediction: scaled_features → scaled_prediction

# Descale back to original space
descaled_prediction = 0.015  # Back to return units (1.5%)

# Convert to prices
price_prediction = base_price * (1 + 0.015)  # Compound return
```


---

## Performance Considerations

### Caching Strategy
```
cache/
  raw_data/           # Yahoo Finance downloads
    CBA.AX_2023-10-10_to_2025-10-10.pkl
  trained_models/     # Saved model weights
    CBA.AX_..._lstm_....keras
```

---

## Available Models Summary

### Deep Learning Models (TensorFlow/Keras)
1. **LSTM** (Long Short-Term Memory)
   - Best for: Capturing long-term dependencies in sequential data
   - Architecture: Stacked LSTM layers with dropout regularization
   - Use case: General-purpose time series prediction

2. **BiLSTM** (Bidirectional LSTM)
   - Best for: Learning patterns from both past and future context
   - Architecture: Bidirectional LSTM layers
   - Use case: When full sequence context improves predictions

3. **GRU** (Gated Recurrent Unit)
   - Best for: Faster training with similar performance to LSTM
   - Architecture: Stacked GRU layers with dropout
   - Use case: Resource-constrained environments

4. **RNN** (Simple Recurrent Neural Network)
   - Best for: Baseline comparison and simple patterns
   - Architecture: Stacked SimpleRNN layers
   - Use case: Quick prototyping and baseline models

5. **CNN-LSTM Hybrid Models**
   - Variants: `cnn_lstm`, `cnn_gru`, `cnn_rnn`, `cnn_bilstm`
   - Best for: Extracting spatial features before temporal modeling
   - Architecture: Conv1D layers → RNN layers → Dense layers
   - Use case: When local patterns in features are important

6. **Attention-based Models**
   - Variants: `attention_lstm`, `attention_gru`, `attention_rnn`, `attention_bilstm`
   - Best for: Focusing on most relevant time steps
   - Architecture: RNN encoder → Multi-head self-attention → Dense layers
   - Use case: When certain time steps (e.g., earnings) are more important

### Statistical Models
7. **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)
   - Best for: Univariate time series with seasonal patterns
   - Architecture: ARIMA with seasonal components
   - Use case: Traditional statistical baseline, interpretable forecasts

### Ensemble Models
8. **SARIMAX + Deep Learning Ensemble**
   - Best for: Combining statistical and neural approaches
   - Architecture: Weighted average of SARIMAX and any TF model
   - Configuration: `--model_name ensemble --ensemble_2 <lstm|gru|rnn|bilstm>`
   - Use case: Leveraging strengths of both paradigms

### Classification Models
9. **Binary Classification Models**
   - Time Series: LSTM, GRU, RNN, BiLSTM with BCE loss
   - Traditional ML: Logistic Regression, Random Forest, SVM
   - Task: Predict up/down movement (binary classification)
   - Approach: Two methods available:
     - BCE-based: Direct binary classification with optimal threshold tuning
     - Price comparison: Regression prediction converted to binary via price comparison

## Extension C.7: Sentiment Analysis Integration

### Overview
The sentiment analysis extension integrates financial news sentiment as additional features to improve stock price predictions. This addresses the limitation that traditional time series models only use historical price data, ignoring market sentiment and news events.

### Architecture

#### Components
1. **News Crawling** ([sentiment/crawl_news.py](dev/sentiment/crawl_news.py))
   - Google News RSS feed integration
   - Yahoo Finance news API
   - Business Today India news
   - Reddit financial subreddits (optional with `--include_social`)
   - Google Trends data (optional with `--include_social`)
   - Date-filtered article fetching
   - Deduplication by URL and (title, date)

2. **Sentiment Analysis** ([sentiment/sentiment_analyzer.py](dev/sentiment/sentiment_analyzer.py))
   - Model: FinBERT (ProsusAI/finbert) - domain-specific BERT for financial text
   - Output: Sentiment scores [-1 to +1] and labels [positive/negative/neutral]
   - Batch processing for efficiency
   - GPU acceleration support

3. **Sentiment Cache** ([sentiment/sentiment_cache.py](dev/sentiment/sentiment_cache.py))
   - Intelligent append-only caching per ticker
   - Cache path: `cache/sentiment/{TICKER}_news.csv`
   - Only fetches missing date ranges
   - Automatic cache validation and coverage checks

#### Data Flow Integration
```
Raw News Articles
    ↓
FinBERT Analysis (ProsusAI/finbert)
    ↓
Daily Sentiment Aggregation (7 features)
    ├─ sentiment_mean: Daily average sentiment
    ├─ news_count: Number of articles per day
    ├─ sentiment_roll3: 3-day rolling average
    ├─ sentiment_roll7: 7-day rolling average
    ├─ sentiment_lag1: Previous day sentiment
    ├─ sentiment_momentum3: 3-day trend direction
    └─ (sentiment features merged BEFORE train/test split)
    ↓
Merge with Stock Data (time-aware)
    ↓
Train/Test Split (prevents data leakage)
    ↓
Model Training with Sentiment Features
```

#### Temporal Alignment Strategy
- News grouped by calendar date (date_only)
- Left join with stock data (preserves all trading days)
- Forward-fill sentiment for days without news (no backward fill to prevent leakage)
- Remaining NaNs filled with 0 (neutral sentiment)

### Implementation Details

#### Feature Engineering
7 temporal sentiment features designed to capture:
1. **Baseline**: Daily mean sentiment, news volume
2. **Short-term trend**: 3-day rolling average
3. **Medium-term trend**: 7-day rolling average
4. **Lag**: Previous day sentiment (temporal dependency)
5. **Momentum**: 3-day cumulative change (trend direction)

#### Usage
```bash
# Enable sentiment features
python main.py --use_sentiment --company AAPL

# Include social media data
python main.py --use_sentiment --include_social --company AAPL

# Classification with sentiment ablation study
python main.py --classification --use_sentiment --company AAPL
```

#### Results and Impact
- **Classification Evaluator**: Automatic ablation study comparing models with/without sentiment
- **Metrics**: Accuracy, Precision, Recall, F1-Score comparison
- **Visualization**:
  - Sentiment vs price overlay plots
  - Sentiment impact analysis charts
  - Confusion matrices with/without sentiment

### Technical Advantages
1. **Domain-Specific**: FinBERT trained on financial text (10K reports, earnings calls)
2. **Efficient Caching**: Only fetches missing data, append-only strategy
3. **No Data Leakage**: Sentiment merged before split, forward-fill only
4. **Rich Features**: 7 engineered features capture temporal patterns
5. **Flexibility**: Works with all model types (LSTM, GRU, RNN, SARIMAX, Ensemble)
6. **Evaluation**: Built-in ablation study for classification tasks

### Performance Considerations
- **Cache Size**: ~1MB per 1000 articles
- **FinBERT Inference**: ~50ms per article on CPU, ~5ms on GPU
- **Social Media**: Reddit adds ~500-1000 posts/ticker, increases fetch time by ~30s

## References

### Key Files
- [main.py](dev/main.py) - CLI interface and orchestration
- [pipeline.py](dev/pipeline.py) - Main data pipeline
- [dataio/converters.py](dev/dataio/converters.py) - Preprocessing
- [dataio/postprocess.py](dev/dataio/postprocess.py) - Postprocessing
- [schemas/bundle.py](dev/schemas/bundle.py) - Data contract
- [model/tf_models.py](dev/model/tf_models.py) - TensorFlow models (9 variants)
- [model/sarimax.py](dev/model/sarimax.py) - SARIMAX model
- [model/ensemble.py](dev/model/ensemble.py) - Ensemble model
- [eval/regression_evaluator.py](dev/eval/regression_evaluator.py) - Regression evaluation
- [eval/classification_evaluator.py](dev/eval/classification_evaluator.py) - Classification evaluation
- [sentiment/sentiment_cache.py](dev/sentiment/sentiment_cache.py) - Sentiment management
- [sentiment/sentiment_analyzer.py](dev/sentiment/sentiment_analyzer.py) - FinBERT integration
- [sentiment/crawl_news.py](dev/sentiment/crawl_news.py) - News fetching

### External Dependencies
- **yfinance**: Yahoo Finance data download
- **pandas**: DataFrame operations
- **numpy**: Array operations
- **scikit-learn**: StandardScaler, metrics, classification models
- **tensorflow/keras**: Deep learning models
- **statsmodels**: SARIMAX implementation
- **pydantic**: Data validation
- **transformers**: FinBERT model (ProsusAI/finbert)
- **torch**: PyTorch backend for FinBERT
- **requests**: HTTP requests for news fetching
