#!/usr/bin/env python3
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("No GPU available. TensorFlow is using CPU.")
else:
    print("GPU is available. TensorFlow is using GPU.")