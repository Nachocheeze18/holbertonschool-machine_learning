#!/usr/bin/env python3
"""
builds a projection block as described in
Deep Residual Learning for Image Recognition (2015):
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    This code defines a function called projection_block
    that implements a building block known as the projection
    block used in deep residual learning for image recognition.
    It performs a series of convolutional operations, batch
    normalization, and element-wise addition to create an activated
    output, following the architecture described in the Deep
    Residual Learning paper.
    """
    F11, F3, F12 = filters

    c1 = K.layers.Conv2D(filters=F11,
                         kernel_size=(1, 1),
                         strides=s,
                         padding='same',
                         kernel_initializer='he_normal'
                         )(A_prev)

    b0 = K.layers.BatchNormalization(axis=3)(c1)

    r0 = K.layers.ReLU()(b0)

    c3 = K.layers.Conv2D(filters=F3,
                         kernel_size=(3, 3),
                         strides=1,
                         padding='same',
                         kernel_initializer='he_normal')(r0)

    b1 = K.layers.BatchNormalization(axis=3)(c3)

    r1 = K.layers.ReLU()(b1)

    c1x1 = K.layers.Conv2D(filters=F12,
                           kernel_size=(1, 1),
                           strides=1,
                           padding='same',
                           kernel_initializer='he_normal')(r1)

    b2 = K.layers.BatchNormalization(axis=3)(c1x1)

    c1x1_short = K.layers.Conv2D(filters=F12,
                                 kernel_size=(1, 1),
                                 strides=s,
                                 padding='same',
                                 kernel_initializer='he_normal')(A_prev)

    batch_short = K.layers.BatchNormalization(axis=3)(c1x1_short)

    add = K.layers.Add()([b2, batch_short])

    r_out = K.layers.ReLU()(add)

    return r_out
