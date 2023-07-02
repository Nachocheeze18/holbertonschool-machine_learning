#!/usr/bin/env python3
"""F1 score"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the true positives, false positives, and false negatives"""
    classes = confusion.shape[0]
    f1_scores = np.zeros(classes)

    for i in range(classes):
        true_positive = confusion[i, i]
        false_positive = np.sum(confusion[:, i]) - true_positive
        false_negative = np.sum(confusion[i, :]) - true_positive

        if true_positive == 0:
            f1_scores[i] = 0.0
        else:
            p = precision(true_positive, false_positive)
            s = sensitivity(true_positive, false_negative)
            f1_scores[i] = 2 * (p * s) / (p + s)

    return f1_scores