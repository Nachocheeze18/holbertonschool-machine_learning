#!/usr/bin/env python3
"""confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """frequency and instances"""
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=int)
    for i in range(labels.shape[0]):
        true = np.argmax(labels[i])
        predicted = np.argmax(logits[i])
        confusion[true][predicted] += 1
    return confusion
