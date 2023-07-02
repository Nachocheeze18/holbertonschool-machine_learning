#!/usr/bin/env python3
"""Layers"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """layer that includes L2 regularization"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(scale=lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, kernel_regularizer=reg)
    output = layer(prev)
    return output
