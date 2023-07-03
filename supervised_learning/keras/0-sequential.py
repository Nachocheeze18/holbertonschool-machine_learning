#!/usr/bin/env python3
"""neural network"""

import tensorflow as tf
from tensorflow import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Keras library"""
    model = keras.Sequential()
    L2_reg = keras.regularizers.l2(lambtha)

    model.add(keras.layers.Dense(layers[0], activation=activations[0],
                                 kernel_regularizer=L2_reg, input_shape=(nx,)))

    for i in range(1, len(layers)):
        model.add(keras.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=L2_reg))
        model.add(keras.layers.Dropout(1 - keep_prob))

    return model
