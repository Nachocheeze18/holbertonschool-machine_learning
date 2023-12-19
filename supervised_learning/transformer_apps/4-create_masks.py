#!/usr/bin/env python3
"""Imports"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def create_masks(inputs, target):
    """all masks for training/validation"""
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(target, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    target_seq_length = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((target_seq_length, target_seq_length)), -1, 0)
    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask