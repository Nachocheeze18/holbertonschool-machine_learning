#!/usr/bin/env python3
"""builds a transition layer as described in Densely
Connected Convolutional Networks:"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ implements a transition layer in a densely connected convolutional
    network by applying batch normalization, ReLU activation, 1x1
    convolution, and 2x2 average pooling to the input tensor, while
    reducing the number of filters and downscaling the spatial dimensions
    of the output."""
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    nb_filters = int(nb_filters * compression)

    X = K.layers.Conv2D(nb_filters, kernel_size=(1, 1), padding='same',
                        kernel_initializer='he_normal')(X)

    X = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    return X, nb_filters
