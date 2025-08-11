# COS30018 - Option C - Task 1: Setup Guide

## Overview
This guide covers the setup and testing of both v0.1 (stock-prediction.py) and P1 (GitHub project) for the stock prediction assignment.

## 1. Environment Setup

### Virtual Environment Creation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Package Installation
```bash
# Install essential packages
pip install -r requirements_clean.txt
```

### Requirements File Details
Our `requirements_clean.txt` contains only essential packages:

**Core Machine Learning:**
- tensorflow>=2.19.0 - Deep learning framework
- keras>=3.11.1 - High-level neural network API
- scikit-learn>=1.7.1 - Machine learning utilities
- numpy>=2.1.3 - Numerical computing
- pandas>=2.3.1 - Data manipulation

**Data Visualization:**
- matplotlib>=3.10.5 - Plotting library

**Stock Data Fetching:**
- yfinance>=0.2.65 - Yahoo Finance data (primary)
- yahoo-fin>=0.8.9.1 - Alternative Yahoo Finance API

**Web Scraping:**
- requests>=2.32.4 - HTTP library
- requests-html>=0.10.0 - HTML parsing
- beautifulsoup4>=4.13.4 - Web scraping
- lxml>=6.0.0 - XML/HTML parser
- lxml_html_clean>=0.4.2 - HTML cleaning

**Jupyter Support:**
- jupyter>=1.0.0 - Notebook interface
- ipykernel>=6.30.1 - Python kernel

**Utilities:**
- python-dateutil>=2.9.0.post0 - Date utilities
- pytz>=2025.2 - Timezone support

## 2. Testing v0.1 (stock-prediction.py)

### Download and Setup
1. Download `stock-prediction.py` from Canvas
2. Place in project directory
3. Ensure virtual environment is activated

### Running v0.1
```bash
python stock-prediction.py
```

### Expected Issues and Fixes
- **Import errors**: Ensure virtual environment is activated
- **API errors**: May need to update data fetching methods
- **Version compatibility**: Use requirements_clean.txt

## 3. Testing P1 (GitHub Project)

### Project Structure
The P1 project (from GitHub) includes:
- `p1.py` - Core functions for data loading and model creation
- `train.py` - Model training script
- `test.py` - Model evaluation script
- `parameters.py` - Configuration parameters
- `p1.ipynb` - Jupyter notebook version

### Running P1
```bash
# Train the model
python train.py

# Test the model
python test.py

# Run Jupyter notebook
jupyter notebook p1.ipynb
```

### Key Features of P1
- LSTM-based stock prediction
- Bidirectional layers support
- Multiple loss functions
- Model checkpointing
- Comprehensive data preprocessing

## 4. Performance Comparison

### Metrics to Compare
1. **Prediction Accuracy**: Mean Absolute Error (MAE)
2. **Training Time**: Time to train models
3. **Model Complexity**: Number of parameters
4. **Data Handling**: Robustness of data loading
5. **Code Quality**: Maintainability and documentation

### Evaluation Criteria
- **Better Prediction**: Lower MAE, higher accuracy
- **Robustness**: Handles API failures gracefully
- **Scalability**: Can handle different stock symbols
- **Maintainability**: Clean, well-documented code

## 5. GitHub Repository Setup

### Repository Structure
```
stock-prediction-project/
├── v0.1/
│   └── stock-prediction.py
├── P1/
│   ├── p1.py
│   ├── train.py
│   ├── test.py
│   ├── parameters.py
│   └── p1.ipynb
├── requirements_clean.txt
├── README.md
└── docs/
    ├── setup_guide.md
    ├── comparison_report.md
    └── weekly_reports/
```

### Wiki Documentation
- **Setup Instructions**: Environment setup guide
- **Weekly Reports**: Progress documentation
- **Comparison Analysis**: v0.1 vs P1 performance
- **Troubleshooting**: Common issues and solutions

## 6. Task 1 Report Requirements

### Report Sections
1. **Environment Setup Summary**
   - Virtual environment creation
   - Package installation details
   - Requirements file explanation

2. **Code Testing Results**
   - Screenshots of successful runs
   - Error handling demonstrations
   - Performance metrics

3. **v0.1 Understanding**
   - Code structure analysis
   - Algorithm explanation
   - Limitations identified

### Submission Checklist
- [ ] Environment setup completed
- [ ] Both v0.1 and P1 tested successfully
- [ ] Performance comparison conducted
- [ ] GitHub repository created
- [ ] Wiki documentation added
- [ ] Task 1 report prepared (PDF)
- [ ] Email notification sent to project leader

## 7. Troubleshooting

### Common Issues
1. **TensorFlow Import Errors**: Ensure virtual environment is activated
2. **API Connection Issues**: Check internet connection and API limits
3. **Version Conflicts**: Use requirements_clean.txt
4. **Memory Issues**: Reduce batch size or sequence length

### Solutions
- Always activate virtual environment before running
- Use `yfinance` as primary data source (more reliable)
- Monitor system resources during training
- Keep requirements file updated

## 8. Next Steps

After completing Task 1:
1. Begin Task 2: Code improvement planning
2. Implement suggested enhancements
3. Document all changes in weekly reports
4. Prepare for final project submission

---

**Note**: This guide should be updated as the project progresses and new issues are discovered.
