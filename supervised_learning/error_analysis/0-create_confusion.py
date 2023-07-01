#!/usr/bin/env python3
"""confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """frequency and instances"""
    num_samples, num_classes = labels.shape
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1

    return confusion_matrix