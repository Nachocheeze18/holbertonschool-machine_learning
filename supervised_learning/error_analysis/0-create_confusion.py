#!/usr/bin/env python3
"""confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """frequency and instances"""
    num_classes = labels.shape[1]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    predicted_labels = np.argmax(logits, axis=1)
    actual_labels = np.argmax(labels, axis=1)
    
    for i in range(len(actual_labels)):
        true_label = actual_labels[i]
        predicted_label = predicted_labels[i]
        confusion_matrix[true_label][predicted_label] += 1
    
    return confusion_matrix
