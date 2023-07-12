#!/usr/bin/env python3
"""LeNet-5 architecture"""
import tensorflow as tf


def lenet5(x, y):
    """image classification, including convolutional and
    fully connected layers, softmax activation, Adam optimization,
    and computes the softmax activated output, loss, and
    accuracy of the network."""

    c1 = tf.layers.conv2d(inputs=x, filters=6, kernel_size=5, padding="same",
                          activation=tf.nn.relu,
                          kernel_initializer=tf.contrib.layers.
                          variance_scaling_initializer())

    p1 = tf.layers.max_pooling2d(inputs=c1,
                                 pool_size=2, strides=2)

    c2 = tf.layers.conv2d(inputs=p1, filters=16, kernel_size=5, padding="valid",
                          activation=tf.nn.relu,
                          kernel_initializer=tf.
                          contrib.layers.variance_scaling_initializer())

    p2 = tf.layers.max_pooling2d(inputs=c2,
                                 pool_size=2, strides=2)

    flat = tf.layers.flatten(p2)

    fc1 = tf.layers.dense(inputs=flat, units=120, activation=tf.nn.relu,
                          kernel_initializer=tf.contrib.layers.
                          variance_scaling_initializer())

    fc2 = tf.layers.dense(inputs=fc1, units=84, activation=tf.nn.relu,
                          kernel_initializer=tf.contrib.layers.
                          variance_scaling_initializer())

    log = tf.layers.dense(inputs=fc2, units=10, activation=None,
                             kernel_initializer=tf.contrib.layers.
                             variance_scaling_initializer())

    s_out = tf.nn.softmax(log)

    loss = tf.reduce_mean(tf.nn.
                          softmax_cross_entropy_with_logits
                          (labels=y, logits=log))

    c_pred= tf.equal(tf.argmax(s_out, 1), tf.argmax(y, 1))

    acc = tf.reduce_mean(tf.cast(c_pred, tf.float32))

    op = tf.train.AdamOptimizer()

    t_op = op.minimize(loss)

    return s_out, t_op, loss, acc
