#!/usr/bin/env python3
"""cost of neural network"""

import tensorflow as tf


def l2_reg_cost(cost):
    """l2 regularization in tensorflow"""
    l2_cost = 0.0
    lambtha = tf.constant(0.01)
    vars = tf.trainable_variables()
    for var in vars:
        l2_cost += tf.nn.l2_loss(var)
    l2_cost *= lambtha
    total_cost = cost + l2_cost

    return total_cost
