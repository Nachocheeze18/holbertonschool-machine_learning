#!/usr/bin/env python3
"""neural network"""

import tensorflow.keras as K



def build_model(nx, layers, activations, lambtha, keep_prob):
    """Keras library"""
    inputs = k.Input(shape=(nx,))
    x = inputs

    for units, activation in zip(layers, activations):
        x = k.layers.Dense(units, activation=activation,
                           kernel_regularizer=k.regularizers.l2(lambtha))(x)
        x = k.layers.Dropout(1 - keep_prob)(x)

    model = k.Model(inputs=inputs, outputs=x)
    return model
