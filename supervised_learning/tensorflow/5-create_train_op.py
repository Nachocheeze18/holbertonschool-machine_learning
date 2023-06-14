#!/usr/bin/env python3
"""loss of the network and learning rate"""
import tensorflow as tf


def create_train_op(loss, alpha):
"""train and loss"""
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
