#!/usr/bin/env python3
"""confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """frequency and instances"""
    num_classes = labels.shape[1]
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for a, p in zip(np.argmax(labels, axis=1), np.argmax(logits, axis=1)):
        matrix[a][p] += 1
    
    return matrix
