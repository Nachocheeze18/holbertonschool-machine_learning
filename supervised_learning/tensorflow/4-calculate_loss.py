#!/usr/bin/env python3
"""Calculate Loss"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculate Loss"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
