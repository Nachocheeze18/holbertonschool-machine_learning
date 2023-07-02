#!/usr/bin/env python3
"""dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates layer using dropout"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    dense = tf.layers.dense(prev, n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=regularizer)
    dropout = tf.layers.dropout(dense, rate=(1-keep_prob))
    return dropout