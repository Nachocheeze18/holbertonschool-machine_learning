#!/usr/bin/env python3
"""create layer"""
import tensorflow as tf


def create_layer(prev, n, activation, i):
    """layer func"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(prev, n, activation=activation,
                            kernel_initializer=init)
    return layer