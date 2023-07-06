#!/usr/bin/env python3
"""optimize"""

import tensorflow.keras as K

def optimize_model(network, alpha, beta1, beta2):
    """Set up Adam optimize and Compile the model with
    categorical crossentropy loss and accuracy metrics"""
    optimizer = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
