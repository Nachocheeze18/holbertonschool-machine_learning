#!/usr/bin/env python3
"""Imports"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import joblib


def preprocess_and_save_data(file_path, window_size):
    """Preprocessing raw data"""
    # Read Bitstamp dataset
    data = pd.read_csv(file_path)

    # Drop NaN values
    df = data.dropna()

    # Keep relevant columns
    df = df[['Close']]

    # Rescale the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # Create sequences and targets
    sequences = []
    targets = []

    # Wrap the loop with tqdm to visualize progress
    for i in tqdm(range(len(data_scaled) - window_size)):
        sequences.append(data_scaled[i:i+window_size])
        targets.append(data_scaled[i+window_size])

    # Make into numpy arrays
    X = np.array(sequences)
    y = np.array(targets)

    # Save data and scaler
    np.save('X.npy', X)
    np.save('y.npy', y)
    joblib.dump(scaler, 'scaler.pkl')

# Specify the file path and window size
file_path = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
window_size = 24

# Preprocess the data and save
preprocess_and_save_data(file_path, window_size)
