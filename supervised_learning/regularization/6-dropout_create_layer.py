#!/usr/bin/env python3
"""dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates layer using dropout"""
    reg = tf.layers.Dropout(rate=keep_prob)
    init = tf.initializers.VarianceScaling(mode="FAN_AVG")

    layer = tf.layers.dense(
        inputs=prev,
        units=n,
        activation=activation,
        kernel_regularizer=reg,
        kernel_initializer=init
    )

    return layer
