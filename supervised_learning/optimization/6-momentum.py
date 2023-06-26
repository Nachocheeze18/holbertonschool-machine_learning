#!/usr/bin/env python3
"""momentum algorithm"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Create momentum"""
    op = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    g = op.compute_gradients(loss)
    m = op.apply_gradients(g)
    return m
