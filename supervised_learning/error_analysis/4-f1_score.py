#!/usr/bin/env python3
"""F1 score"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calulates f1 score for each class"""
    classes = confusion.shape[0]
    f1_scores = np.zeros(classes)

    for i in range(classes):
        tp = confusion[i, i]
        fp = np.sum(confusion[:, i]) - tp
        fn = np.sum(confusion[i, :]) - tp

        precision = precision(tp, fp)
        sensitivity = sensitivity(tp, fn)

        if precision + sensitivity > 0:
            f1_scores[i] = 2 * (precision * sensitivity) / (precision + sensitivity)

    return f1_scores