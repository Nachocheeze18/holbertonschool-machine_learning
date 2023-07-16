#!/usr/bin/env python3
"""inception block"""
import tensorflow as tf


def inception_block(A_prev, filters):
    """This function uses the Keras API from TensorFlow
    to create the different convolutional layers of the
    inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution branch
    c1 = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding='same', activation='relu')(A_prev)

    # 1x1 convolution followed by 3x3 convolution branch
    c3_reduce = tf.keras.layers.Conv2D(filters=F3R, kernel_size=(1, 1), padding='same', activation='relu')(A_prev)
    c3 = tf.keras.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same', activation='relu')(c3_reduce)

    # 1x1 convolution followed by 5x5 convolution branch
    c5_reduce = tf.keras.layers.Conv2D(filters=F5R, kernel_size=(1, 1), padding='same', activation='relu')(A_prev)
    c5 = tf.keras.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same', activation='relu')(c5_reduce)

    # Max pooling followed by 1x1 convolution branch
    maxpool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(A_prev)
    maxpool_proj = tf.keras.layers.Conv2D(filters=FPP, kernel_size=(1, 1), padding='same', activation='relu')(maxpool)

    # Concatenate the outputs of all branches
    inception_output = tf.keras.layers.concatenate([c1, c3, c5, maxpool_proj], axis=3)

    return inception_output
