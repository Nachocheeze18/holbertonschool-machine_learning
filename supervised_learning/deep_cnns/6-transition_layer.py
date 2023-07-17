#!/usr/bin/env python3
""""""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Function that builds a transition layer"""
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    nb_filters = int(nb_filters * compression)

    X = K.layers.Conv2D(nb_filters, kernel_size=(1, 1), padding='same',
                        kernel_initializer='he_normal')(X)

    X = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    return X, nb_filters
