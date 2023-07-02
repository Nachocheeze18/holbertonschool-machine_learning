#!/usr/bin/env python3
"""cost of neural network"""

import tensorflow as tf


def l2_reg_cost(cost):
    """l2 regularization in tensorflow"""
    loss = tf.losses.get_regularization_losses()
    total_cost = cost + tf.reduce_sum(loss)
    return total_cost

