#!/usr/bin/env python3
"""Imports"""
import numpy as np


def variance(X, C):
    """
    Function that calculates the total intra-cluster variance
    X: numpy.ndarray shape (n, d) containing the dataset
    C: numpy.ndarray shape (k, d) containing centroid means for each cluster
    Returns var, or None on failure
        var is the total variance
    """
    # Guard against bad input data
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None

    try:
        min_dist = np.min(np.linalg.norm((X[:, np.newaxis, :] - C), axis=2), axis=1)
        var = np.sum(min_dist ** 2)

        return var
    except Exception:
        return None