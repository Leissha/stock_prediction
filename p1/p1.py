import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input # type: ignore
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_handling import shuffle_in_unison

import numpy as np
import pandas as pd
import random

# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


def get_stock_data(ticker):
    """
    Get stock data using yfinance (modern version of yahoo_fin).
    """
    # Default to 2 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if data is not None and not data.empty:
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten the column names
            # yfinance sometimes returns data with multi-level col names like:
            # ('Close', 'AAPL')
            # ('Volume', 'AAPL')
            # ('High', 'AAPL')
            # => We want to flatten them to: ['close', 'volume', 'open', 'high', 'low']
            data.columns = [col[0].lower() if col[1] == ticker else f"{col[0].lower()}_{col[1].lower()}" for col in data.columns]
            # Rename 'close' to 'adjclose' to match expected format
            if 'close' in data.columns:
                data = data.rename(columns={'close': 'adjclose'})
        
        print(f"Successfully loaded {ticker} data using yfinance")
        return data
    else:
        raise ValueError(f"Failed to load data for {ticker}")


def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
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
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it using yfinance (more reliable than yahoo_fin)
        df = get_stock_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

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

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # append to df the target column (label) by shifting by `lookup_step`
        # i.e: n_steps (look back) = 2, lookup_step (predict ahead) = 2
        # Before:             
        #               adjclose  volume       date
        #   2023-01-01       100    1000 2023-01-01
        #   2023-01-02       102    1200 2023-01-02

        # After:             
        #                   adjclose  future     date      
        #   2023-01-01       100       102    2023-01-01   
        #   2023-01-02       102       105    2023-01-02   
        #   2023-01-03       105       103    2023-01-03   
        #   2023-01-04       103       107    2023-01-04   
        #   2023-01-05       107       NaN    2023-01-05   
        #   2023-01-06       110       NaN    2023-01-06   
    df['future'] = df['adjclose'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs 
        # in i.e: store last_seq =  [[107 1100 120 112 113] # other features.values like volume, open, high, low
        #                            [110 1230 130 122 123]] )
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    
    # drop NaNs
        # After:
        #                   adjclose  future     date     
        #   2023-01-01       100       102    2023-01-01  
        #   2023-01-02       102       105    2023-01-02  
        #   2023-01-03       105       103    2023-01-03  
        #   2023-01-04       103       107    2023-01-04  
    df.dropna(inplace=True)

    sequence_data = []
    # effective sliding window: push old data
    # automatically sizing keeps exactly n_steps (50 days) data 
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)           # Add current day's data
        if len(sequences) == n_steps:     # When we have 50 days
            sequence_data.append([np.array(sequences), target])  # Save as one training example

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # After splitting, shuffle batch order (sequence_data) within each set (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # split the dataset randomly (mix up all sequence)
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)

    # get the list of test set dates 
        # result["X_test"] shape: (test_samples, time_steps, features)
        # [:, -1, -1] means: all samples, last time step, last feature (which is the date)
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result


def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    # Add Input layer as the first layer for new tensorflow
    model.add(Input(shape=(sequence_length, n_features)))
    
    for i in range(n_layers):
        if i == 0:
            # first layer (no input_shape needed since we have Input layer)
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model