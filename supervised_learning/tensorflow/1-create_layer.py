#!/usr/bin/env python3
"""create layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
"""layer func"""
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    input_shape = prev.get_shape().as_list()[1:]
    weights = tf.Variable(initializer([input_shape[-1], n]), name='layer_weights')
    biases = tf.Variable(tf.zeros([n]), name='layer_biases')
    layer_output = tf.matmul(prev, weights) + biases
    output = activation(layer_output)
    return output
