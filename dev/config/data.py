TICKER = 'CBA.AX'
# TRAIN_START = '2021-01-01'
# TRAIN_END = '2025-08-19'  # Use all available data
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=730)
START_DATE = start_date.strftime('%Y-%m-%d')
END_DATE = end_date.strftime('%Y-%m-%d')

# Price value can be "Close", "Open", "High", "Low", "AdjClose", "Volume"
TARGET_FEATURE = "Close"  
LAG_DAYS = 60

# Data processing
LOOKUP_STEP = 1
TEST_SIZE = 0.2
VAL_SIZE = 0.2
SHUFFLE = True
SCALE = True
SPLIT_METHOD = 'date'
RANDOM_STATE = 42

# model
MODEL_NAME='lstm'
LOSS = "mean_squared_error"
LAYERS = [50,50,50]
DROPOUT = 0.2
OPTIMIZER = "adam"
BATCH_SIZE = 32
EPOCHS = 50
