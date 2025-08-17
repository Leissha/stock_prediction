# Stock Prediction Project - COS30018 Option C

## Overview
This repository contains the implementation and analysis of stock prediction models for COS30018 Option C assignment. The project compares and improves upon two different approaches to stock price prediction using LSTM neural networks.

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

### 2. Run the Models
```bash
# Test v0.1 (Original YouTube tutorial - fixed)
cd v0.1
python v0.1.py

# Test P1 (GitHub project)
cd p1
python train.py  # Train the model
python test.py   # Evaluate the model
# or notebook
run p1.ipynb
```

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
├── dev/                    # Development versions (OOP approach)
│   └── enhanced_v0.1.py   # Object-oriented enhancement (in development)
├── requirements_clean.txt  # Essential package dependencies
├── README.md              # Project overview
└── docs/
    └── TASK_1_REPORT.md   # This comprehensive report
```

## Performance Comparison
| Metric | v0.1 | P1 |
|--------|------|----|
| Prediction Error | 7.9% | ~1.2% |
| Features | 1 (Close) | 5 (OHLCV) |
| Trading Strategy | None | Buy/Sell signals |
| Code Quality | Basic | Modular |

**Result: P1 significantly outperforms v0.1**

## Documentation
- **[Task 1 Report](docs/TASK_1_REPORT.md)** - Complete setup, testing, and analysis report
- **GitHub Wiki** - Will be used for Tasks 2-7 documentation

## Requirements
| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **Core ML** | tensorflow | ≥2.19.0 | Deep learning framework |
| | keras | ≥3.11.1 | High-level neural network API |
| | scikit-learn | ≥1.7.1 | Machine learning utilities |
| | numpy | ≥2.1.3 | Numerical computing |
| | pandas | ≥2.3.1 | Data manipulation |
| **Visualization** | matplotlib | ≥3.10.5 | Plotting library |
| **Data Fetching** | yfinance | ≥0.2.65 | Yahoo Finance data (primary) |
| | yahoo-fin | ≥0.8.9.1 | Alternative Yahoo Finance API |
| **Utilities** | requests | ≥2.32.4 | HTTP library |
| | jupyter | ≥1.0.0 | Notebook interface |

## Status
- ✅ **Task 1 Complete**: Environment setup, code testing, performance comparison
- 🚧 **Task 2 in progress**: Refactoring code for future extendability and modifiability.

---

**Last Updated**: August 17, 2025  
**Course**: COS30018 Option C