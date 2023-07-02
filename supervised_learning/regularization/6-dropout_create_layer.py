#!/usr/bin/env python3
"""dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates layer using dropout"""
    w = tf.Variable(tf.random.normal([prev.shape[1], n]))
    b = tf.Variable(tf.zeros([n]))
    lin_out = tf.matmul(prev, w) + b
    act_out = activation(lin_out)
    dropout_out = tf.nn.dropout(act_out, keep_prob)
    return dropout_out
