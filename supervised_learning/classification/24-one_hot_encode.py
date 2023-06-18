#!/usr/bin/env python3
"""one hot encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """numeric label to matrix"""
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < np.max(Y) + 1:
        return None
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot