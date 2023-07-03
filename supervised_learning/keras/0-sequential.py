#!/usr/bin/env python3
"""neural network"""

import tensorflow.keras as K



def build_model(nx, layers, activations, lambtha, keep_prob):
    """Keras library"""
    model = k.models.Sequential()

    model.add(k.layers.Dense(layers[0], activation=activations[0],
                             input_shape=(nx,),
                             kernel_regularizer=k.regularizers.l2(lambtha)))

    for units, activation in zip(layers[1:], activations[1:]):
        model.add(k.layers.Dropout(1 - keep_prob))
        model.add(k.layers.Dense(units, activation=activation,
                                 kernel_regularizer=k.regularizers.l2(lambtha)))

    return model
