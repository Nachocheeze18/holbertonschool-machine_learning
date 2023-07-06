#!/usr/bin/env python3
"""neural network"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Keras library"""
    model = K.Sequential()

    model.add(K.layers.InputLayer(input_shape=(nx,)))

    for layer, units in enumerate(layers):
        model.add(K.layers.Dense(units=units,
                                 activation=activations[layer],
                                 kernel_regularizer=K.regularizers.l2(lambtha)
                                 ))

        if layer < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
