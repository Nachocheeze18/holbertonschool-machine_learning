#!/usr/bin/env python3
"""Imports"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None

    n, d = X.shape
    cen = np.zeros((k, d))

    cen[0] = X[np.random.choice(n, 1)]

    for i in range(1, k):
        dist = np.array([min(np.linalg.norm(x - c) ** 2 for c in cen[:i]) for x in X])
        prob = dist / dist.sum()
        cen[i] = X[np.random.choice(n, 1, p=prob)]

    return cen