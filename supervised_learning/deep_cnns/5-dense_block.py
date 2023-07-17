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
        bottleneck = K.layers.BatchNormalization()(X)
        bottleneck = K.layers.Activation('relu')(bottleneck)
        bottleneck = K.layers.Conv2D(
            inter_channel, (1, 1), padding='same',
            kernel_initializer='he_normal')(bottleneck)

        conv_layer = K.layers.BatchNormalization()(bottleneck)
        conv_layer = K.layers.Activation('relu')(conv_layer)
        conv_layer = K.layers.Conv2D(
            growth_rate, (3, 3), padding='same',
            kernel_initializer='he_normal')(conv_layer)

        X = K.layers.Concatenate()([X, conv_layer])

        nb_filters += growth_rate

        i += 1

    return X, nb_filters
