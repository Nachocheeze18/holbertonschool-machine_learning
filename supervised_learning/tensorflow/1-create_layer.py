#!/usr/bin/env python3
"""create layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """layer func"""
    with tf.variable_scope("layer"):
        initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        weih = tf.get_variable("weights", shape=[prev.get_shape()[1], n], initializer=initializer)
        biases = tf.get_variable("biases", shape=[n], initializer=tf.zeros_initializer())

        layer = tf.matmul(prev, weights) + biases

        if activation is not None:
            layer = activation(layer)

    return layer