#!/usr/bin/env python3
"""confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """frequency and instances"""
    num_classes = labels.shape[1]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(labels.shape[0]):
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(logits[i])
        confusion_matrix[true_label][predicted_label] += 1
    return confusion_matrix
