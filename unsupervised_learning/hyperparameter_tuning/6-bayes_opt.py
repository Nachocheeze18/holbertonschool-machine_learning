#!/usr/bin/env python3
"""Imports"""
import numpy as np
import GPyOpt
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os

data = load_iris()
X = data.data
y = (data.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(learning_rate, num_units, dropout_rate, l2_weight):
    """Define the neural network model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def objective_function(params):
    """Define the objective function to optimize"""
    learning_rate, num_units, dropout_rate, l2_weight, batch_size = params[0]
    
    model = create_model(learning_rate, num_units, dropout_rate, l2_weight)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    
    model.save(f"model_lr{learning_rate}_units{num_units}_dropout{dropout_rate}_l2{l2_weight}_batch{batch_size}.h5")

    return -f1

bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.001, 0.1)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (16, 32, 64)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (0.0, 0.1)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)}
]

optimizer = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=bounds, initial_design_numdata=5)

max_iter = 30
optimizer.run_optimization(max_iter)

with open('bayes_opt.txt', 'w') as file:
    file.write(f'Optimal Parameters: {optimizer.x_opt}\n')
    file.write(f'Optimal F1 Score: {optimizer.fx_opt}\n')

optimizer.plot_convergence()

for filename in os.listdir('.'):
    if filename.startswith('model_lr') and filename.endswith('.h5'):
        os.remove(filename)
