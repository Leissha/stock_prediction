import os
import numpy as np
from loguru import logger
from collections import deque
from sklearn import preprocessing

# Import existing modular code
from data_preprocessing.data_splitting import split_data
from data_preprocessing.data_loading import load_stock_data
from utils.file_handling import ensure_directory_exists


class DataProcessor:
    """
    Comprehensive data processing class that meets Task 2 requirements.
    
    This class provides a clean interface for loading and processing stock data with:
    - Multiple feature support (Open, High, Low, Close, Volume, AdjClose)
    - Flexible data splitting methods (date-based, ratio-based, random)
    - Proper feature scaling with scaler storage
    - Local data caching for efficiency
    - NaN handling with forward/backward fill
    - Sequence creation for LSTM models
    
    The class reuses existing modular code where applicable while adding
    the missing functionality required by Task 2.
    """
    
    def __init__(self, cache_dir='dev/cache'):
        """
        Initialize the DataProcessor.
        
        Args:
            cache_dir (str): Directory for caching processed data
        """
        self.cache_dir = ensure_directory_exists(cache_dir)
        self.scalers = {}
        self.data = None
        self.processed_data = None

    
    def data_processing(self, start_date, end_date, ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, splitting_method="date",
                    test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
        """
        Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
        Params:
            ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
            n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
            scale (bool): whether to scale prices from 0 to 1, default is True
            shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
            lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
            split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
                to False will split datasets in a random way
            test_size (float): ratio for test data, default is 0.2 (20% testing data)
            feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yfinance
        """
        df = load_stock_data(ticker, start_date, end_date, cache_dir=os.path.join(self.cache_dir, 'raw_data'))
        logger.info(f"\nRaw data: {df.head()}")
        
        # this will contain all the elements we want to return from this function
        result = {}
        # we will also return the original dataframe itself
        result['df'] = df.copy()

        # make sure that the passed feature_columns exist in the dataframe
        for col in feature_columns:
            assert col in df.columns, f"'{col}' does not exist in the dataframe."

        # add date as a column
        if "date" not in df.columns:
            df["date"] = df.index
        logger.info(f"\nDate column: {df['date'].head()}")
        
        if scale:
            column_scaler = {}
            # scale the data (prices) from 0 to 1
            for column in feature_columns:
                scaler = preprocessing.MinMaxScaler()
                df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
                column_scaler[column] = scaler

            logger.info(f"\nScaled data: \n{df.head()}")
            # add the MinMaxScaler instances to the result returned
            result["column_scaler"] = column_scaler
            logger.info(f"\nColumn scaler: \n{result['column_scaler']}")

        # add the target column (label) by shifting by `lookup_step`
        df['future'] = df['Close'].shift(-lookup_step)
        logger.info(f"\nFuture column (last 8 rows): \n{df['future'].tail(8)}")

        # last `lookup_step` columns contains NaN in future column
        # get them before droping NaNs
        last_sequence = np.array(df[feature_columns].tail(lookup_step))
        logger.info(f"\nLast sequence: \n{last_sequence}")
        
        # drop NaNs
        df.dropna(inplace=True)
        logger.info(f"\nDropped NaNs: \n{df.head()}")
        
        sequence_data = []
        sequences = deque(maxlen=n_steps)
        
        for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])
        logger.info(f"\nSequence data: \n{sequence_data[:1][:1]}")
        
        # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
        # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
        # this last_sequence will be used to predict future stock prices that are not available in the dataset
        last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)
        # add to result
        result['last_sequence'] = last_sequence
        logger.info(f"\nLast sequence: \n{result['last_sequence'].shape}")
        
        # construct the X's and y's
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)
        logger.info(f"\nX: \n{X[:1][:1]}")
        logger.info(f"\ny: \n{y[:1][:1]}")
        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Return result[X_train], result[X_test], result[y_train], result[y_test]
        result = split_data(df=result, X=X, y=y, method=splitting_method, test_size=test_size, shuffle=shuffle)
        
        # replace the bad log
        logger.info(f"\nResult after splitting: \nX_train={result['X_train'].shape}, \nX_test={result['X_test'].shape}, \ny_train={result['y_train'].shape}, \ny_test={result['y_test'].shape}")
        
        # get the list of test set dates
        dates = result["X_test"][:, -1, -1]
        # retrieve test features from the original dataframe
        result["test_df"] = result["df"].loc[dates]
        # remove duplicated dates in the testing dataframe
        result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
        # remove dates from the training/testing sets & convert to float32
        result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
        logger.info(f"\n Inspect data: X_train range: {np.max(result['X_train'])} - {np.min(result['X_train'])} \nX_test max {np.max(result['X_test'])} - {np.min(result['X_test'])} \ny_train range {np.max(result['y_train'])} - {np.min(result['y_train'])} \ny_test range {np.max(result['y_test'])} - {np.min(result['y_test'])}")

        return result
