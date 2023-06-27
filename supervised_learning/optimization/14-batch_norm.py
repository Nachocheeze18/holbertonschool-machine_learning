#!/usr/bin/env python3
"""neural network"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """layer batch"""
    layer = tf.layers.Dense(units=n, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    z = layer(prev)
    batch = tf.layers.batch_normalization(z, epsilon=1e-8, trainable=True)
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    z = gamma * batch + beta
    output = activation(z)
    return output
