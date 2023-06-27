#!/usr/bin/env python3
"""neural network"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """layer batch"""
    cont = tf.compat.v1.keras.initializers.VarianceScaling(mode="fan_avg")
    den_lay = tf.compat.v1.layers.dense(inputs=prev, units=n, kernel_initializer=cont)
    mean, var = tf.nn.moments(den_lay, axes=0)
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    epsilon = 1e-8
    Z = tf.nn.batch_normalization(
        den_lay, mean, var, beta, gamma, epsilon
    )
    return activation(Z)