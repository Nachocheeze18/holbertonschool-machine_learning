#!/usr/bin/env python3
"""confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """frequency and instances"""
    lab = labels.shape[1]
    log = logits.shape[1]

    matrix = np.zeros((lab, log))
    prediction = np.argmax(logits, axis=1)
    actual = np.argmax(labels, axis=1)
    for i in range(len(actual)):
        a = actual[i]
        predict = prediction[i]
        matrix[a][predict] += 1

    return matrix
