#!/usr/bin/env python3
"""Imports"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib


def load_and_preprocess_data(file_path):
    """Load Preprocessed data"""
    # Read data
    data = pd.read_csv(file_path)

    # Drop NaN values
    df = data.dropna()

    # Keep relevant columns
    df = df[['Close']]

    return df

def preprocess_data(data, seq_length):
    """rescale preprocessed data"""
    # Rescale the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Create sequences and targets
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)

    # Make into numpy arrays
    X = np.array(sequences)
    y = np.array(targets)

    return X, y, scaler

def train_model(X_train, y_train, X_val, y_val, seq_length):
    """train the model"""
    # Define the RNN model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='tanh', input_shape=(seq_length, 1)),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Create a tf.data.Dataset
    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    # Train the model
    model.fit(train_dataset, epochs=10, validation_data=val_dataset)

    return model

def evaluate_and_save_model(model, X_test, y_test, scaler, model_save_path):
    """evaluate and save the rnn model"""
    # Evaluate the model on test data
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')

    # Save the trained model
    model.save(model_save_path)

    # Load the scaler used during preprocessing
    scaler = joblib.load('scaler.pkl')

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Inverse transform the scaled data to the original scale
    y_test_actual = scaler.inverse_transform(y_test)
    y_pred_actual = scaler.inverse_transform(y_pred)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, label='Actual', color='blue')
    plt.plot(y_pred_actual, label='Predicted', color='red')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Load and preprocess data
file_path = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
df = load_and_preprocess_data(file_path)

# Split the data into training, validation, and test sets
train_size = int(len(df) * 0.8)
val_size = int(len(df) * 0.1)
train_data, val_data, test_data = df[:train_size], df[train_size:train_size + val_size], df[train_size + val_size:]

# Create sequences and targets for training, validation, and test sets
seq_length = 24
X_train, y_train, scaler = preprocess_data(train_data, seq_length)
X_val, y_val, _ = preprocess_data(val_data, seq_length)
X_test, y_test, _ = preprocess_data(test_data, seq_length)

# Train the model
model = train_model(X_train, y_train, X_val, y_val, seq_length)

# Save the model
model_save_path = 'btc_forecasting_model.h5'
evaluate_and_save_model(model, X_test, y_test, scaler, model_save_path)
