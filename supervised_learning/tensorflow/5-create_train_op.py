#!/usr/bin/env python3
"""loss of the network and learning rate"""
import tensorflow as tf


def create_train_op(loss, alpha):
"""train and loss"""
   return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
