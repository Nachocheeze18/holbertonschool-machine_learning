#!/usr/bin/env python3
"""builds a dense block as described in
Densely Connected Convolutional Networks:"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """builds a dense block, as described in the "Densely
    Connected Convolutional Networks" paper. The function
    takes an input tensor, applies a series of batch normalization,
    activation, and convolutional layers, and concatenates the
    output of each layer with the input tensor to create a dense
    connectivity pattern."""
    i = 0
    while i < layers:
        inter_channel = 4 * growth_rate
        b1 = K.layers.BatchNormalization()(X)
        b1 = K.layers.Activation('relu')(b1)
        b1 = K.layers.Conv2D(
            inter_channel, (1, 1), padding='same',
            kernel_initializer='he_normal')(b1)

        c1 = K.layers.BatchNormalization()(b1)
        c1 = K.layers.Activation('relu')(c1)
        c1 = K.layers.Conv2D(
            growth_rate, (3, 3), padding='same',
            kernel_initializer='he_normal')(c1)

        X = K.layers.Concatenate()([X, c1])

        nb_filters += growth_rate

        i += 1

    return X, nb_filters
