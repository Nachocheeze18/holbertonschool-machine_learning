#!/usr/bin/env python3
"""builds an identity block as described in
Deep Residual Learning for Image Recognition (2015):"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """This code defines an identity block, which is a building block for
    deep residual networks used in image recognition. It performs a series
    of convolutions, batch normalization, and element-wise addition with
    the input, followed by ReLU activation, to create a residual
    connection that allows the network to learn residual mappings."""
    F11, F3, F12 = filters

    init = K.initializers.he_normal()

    c1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=1, padding='same',
                            kernel_initializer=init)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(c1)
    r1 = K.layers.ReLU()(bn1)

    c2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), strides=1, padding='same',
                            kernel_initializer=init)(r1)
    bn2 = K.layers.BatchNormalization(axis=3)(c2)
    r2 = K.layers.ReLU()(bn2)

    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=1, padding='same',
                            kernel_initializer=init)(r2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    add = K.layers.Add()([bn3, A_prev])
    r_out = K.layers.ReLU()(add)

    return r_out
