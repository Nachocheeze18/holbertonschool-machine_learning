#!/usr/bin/env python3
"""keras object"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """he function creates and returns a Keras Model object using
    K.Model with the input layer as the input and the last dense
    layer as the output. The resulting model represents the neural
    network architecture defined by the provided parameters."""
    inputs = K.Input(shape=(nx,))
    x = K.layers.Dense(units=layers[0],
                       activation=activations[0],
                       kernel_regularizer=K.regularizers.l2(lambtha)
                       )(inputs)

    for idx, units in enumerate(layers[1:], start=1):
        dropout_layer = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(units=units,
                           activation=activations[idx],
                           kernel_regularizer=K.regularizers.l2(lambtha)
                           )(dropout_layer)

    model = K.Model(inputs=inputs, outputs=x)
    return model
