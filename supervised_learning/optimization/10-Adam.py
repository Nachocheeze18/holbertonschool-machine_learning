#!/usr/bin/env python3
"""train with adam algorithm"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """neural network"""
    s = tf.Variable(0, trainable=False, name='global_step')
    op = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                beta2=beta2, epsilon=epsilon)
    t = op.minimize(loss, global_step=s)
    return t
