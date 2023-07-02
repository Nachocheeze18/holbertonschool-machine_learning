#!/usr/bin/env python3
"""dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates layer using dropout"""
    reg = tf.keras.layers.Dropout(rate=keep_prob)
    init = tf.keras.initializers.VarianceScaling(mode="fan_avg")

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=reg,
        kernel_initializer=init
    )(prev)

    return layer
