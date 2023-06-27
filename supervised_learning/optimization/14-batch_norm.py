#!/usr/bin/env python3
"""neural network"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """layer batch"""
    dense_layer = tf.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    )
    dense_output = dense_layer(prev)
    gamma = tf.Variable(tf.ones([n]), name="gamma")
    beta = tf.Variable(tf.zeros([n]), name="beta")
    mean, variance = tf.nn.moments(dense_output, axes=[0])
    normalized_output = tf.nn.batch_normalization(
        dense_output,
        mean,
        variance,
        beta,
        gamma,
        1e-8
    )
    activated_output = activation(normalized_output) if activation is not None else normalized_output
    return activated_output
