#!/usr/bin/env python3
"""inception block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """This function uses the Keras API from TensorFlow
    to create the different convolutional layers of the
    inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters

    c1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                         padding='same', activation='relu')(A_prev)

    c3_reduce = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1),
                                padding='same', activation='relu')(A_prev)

    c3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                         padding='same', activation='relu')(c3_reduce)

    c5_reduce = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                                padding='same', activation='relu')(A_prev)

    c5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5),
                         padding='same', activation='relu')(c5_reduce)

    maxpool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    strides=(1, 1), padding='same')(A_prev)

    maxpool_proj = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                                   padding='same', activation='relu')(maxpool)

    incept_out = K.layers.concatenate([c1, c3, c5,
                                       maxpool_proj], axis=3)

    return incept_out
