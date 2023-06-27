#!/usr/bin/env python3
"""neural network"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """layer batch"""
    layer = tf.layers.Dense(units=n, activation=None,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    output = layer(prev)

    gamma = tf.Variable(tf.ones([n]), name="gamma")
    beta = tf.Variable(tf.zeros([n]), name="beta")

    mean, variance = tf.nn.moments(output, axes=[0])

    normal_op = tf.nn.batch_normalization
    (output, mean, variance, beta, gamma, 1e-8)

    activate_op = activation(normal_op)if activation is not None else normal_op

    return activate_op