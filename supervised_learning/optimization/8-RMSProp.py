#!/usr/bin/env python3
"""RMSProp"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """RMSProp optimization"""
    op = tf.train.RMSPropOptimizer(learning_rate=alpha, 
                                   decay=beta2, epsilon=epsilon)
    t = op.minimize(loss)
    return t
