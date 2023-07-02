#!/usr/bin/env python3
"""dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates layer using dropout"""
    init = tf.initializers.VarianceScaling(mode="FAN_AVG")
    reg = tf.keras.regularizers.l2(0.1)
    
    dense = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=init,
                                  kernel_regularizer=reg)(prev)
    
    dropout = tf.keras.layers.Dropout(1 - keep_prob)(dense)
    
    return dropout
