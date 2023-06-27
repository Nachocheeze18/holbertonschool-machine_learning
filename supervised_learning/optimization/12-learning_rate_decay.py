#!/usr/bin/env python3
"""learning rate decay operation"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """inverse time decay"""
    learn = tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True
    )
    return learn
