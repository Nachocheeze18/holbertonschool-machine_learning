#!/usr/bin/env python3
"""one hot encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """numeric label to matrix"""
    if type(Y) is not np.ndarray:
        return None

    try:
        m = Y.shape[0]
        one_hot_matrix = np.zeros((classes, m))

        for i in range(m):
            if Y[i] >= 0 and Y[i] < classes:
                one_hot_matrix[Y[i], i] = 1

        return one_hot_matrix

    except Exception:
        return None

