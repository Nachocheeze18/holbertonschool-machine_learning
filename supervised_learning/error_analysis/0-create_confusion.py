#!/usr/bin/env python3
"""confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """frequency and instances"""
    l1 = labels.shape[1]
    l2 = logits.shape[1]

    matrix = np.zeros((l1, l2))

    prediction = np.argmax(logits, axis=1)
    actual = np.argmax(labels, axis=1)

    matrix[np.arange(len(actual)), prediction] += 1

    return matrix
