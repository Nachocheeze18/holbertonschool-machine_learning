#!/usr/bin/env python3
"""cost of neural network"""

import tensorflow as tf


def l2_reg_cost(cost):
    """l2 regularization in tensorflow"""
    lambtha = tf.constant(0.01)  # regularization parameter
    l2_cost = lambtha * tf.reduce_sum
    ([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    total_cost = cost + l2_cost
    return total_cost
