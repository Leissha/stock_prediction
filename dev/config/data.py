COMPANY = 'CBA.AX'
# TRAIN_START = '2021-01-01'
# TRAIN_END = '2025-08-19'  # Use all available data
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=730)
TRAIN_START = start_date.strftime('%Y-%m-%d')
TRAIN_END = end_date.strftime('%Y-%m-%d')
# Price value can be "Close", "Open", "High", "Low", "AdjClose", "Volume"
PRICE_VALUE = "Close"  
LAG_DAYS = 60

# model
N_STEPS = 60
FEATURE_COLUMNS = ['close']
LOSS = "mean_squared_error"
UNITS = 50
N_LAYERS = 3
DROPOUT = 0.2
OPTIMIZER = "adam"
BATCH_SIZE = 32
EPOCHS = 50
TEST_SIZE = 0.2