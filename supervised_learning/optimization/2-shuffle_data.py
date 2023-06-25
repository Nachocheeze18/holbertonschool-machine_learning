#!/usr/bin/env python3
"""shuffles data points"""
import numpy as np


def shuffle_data(X, Y):
    """shuffle func"""
    assert X.shape[0] == Y.shape[0]
    m = X.shape[0]
    perm = np.random.perm(m)
    _X = X[perm]
    _Y = Y[perm]
    return _X, _Y
