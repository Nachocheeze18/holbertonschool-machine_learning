#!/usr/bin/env python3
"""neural network"""

import tensorflow.keras as K



def build_model(nx, layers, activations, lambtha, keep_prob):
    """Keras library"""
    model = K.Sequential()
    L2_reg = K.regularizers.l2(lambtha)

    model.add(K.layers.Dense(layers[0], activation=activations[0],
                             kernel_regularizer=L2_reg, input_shape=(nx,)))

    for i in range(1, len(layers)):
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=L2_reg))
        model.add(K.layers.Dropout(1 - keep_prob))

    return model
