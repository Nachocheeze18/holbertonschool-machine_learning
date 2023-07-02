#!/usr/bin/env python3
""""""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ layer that includes L2 regularization"""
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularize = tf.contrib.layers.l2_regularizer(scale=lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initialize, kernel_regularizer=regularize)
    output = layer(prev)
    return output
