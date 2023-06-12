#!/usr/bin/env python3
"""create layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """layer func"""
     initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    activation_name = activation.__name__ if activation is not None else ""
    layer_name = "layer" + "_" + str(i) + activation_name

    layer = tf.layers.dense(inputs=prev,
                            units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name=layer_name)

    return layer
