#!/usr/bin/env python3
"""sensetivity"""

import numpy as np


def sensitivity(confusion):
    """calculates sensitivity for a class"""
    classes = confusion.shape[0]
    sensitivity_values = np.zeros(classes)

    for i in range(classes):
        true = confusion[i, i]
        actual_positives = np.sum(confusion[i, :])

        sensitivity_values[i] = true / actual_positives

    return sensitivity_values
