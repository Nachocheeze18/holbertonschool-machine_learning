#!/usr/bin/env python3
"""cost of neural network"""

import tensorflow as tf


def l2_reg_cost(cost):
    """l2 regularization in tensorflow"""
    return (cost + tf.reduce_sum(tf.losses.get_regularization_losses()))
