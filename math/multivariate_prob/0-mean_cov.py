#!/usr/bin/env python3
"""Imports"""
import numpy as np


def mean_cov(X):
    """calculates the mean and covariance of a data set"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")
    
    mean = np.mean(X, axis=0, keepdims=True)

    cov = np.zeros((d, d))

    for i in range(d):
        for j in range(i, d):
            cov[i, j] = np.sum((X[:, i] - mean[:, i]) * (X[:, j] - mean[:, j])) / (n - 1)
            cov[j, i] = cov[i, j]

    return mean, cov