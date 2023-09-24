#!/usr/bin/env python3
"""Imports"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means"""
    try:
        if k <= 0 or not isinstance(k, int) or X.shape[0] < k:
            return None

        n, d = X.shape

        min_values = X.min(axis=0)
        max_values = X.max(axis=0)

        cen = np.random.uniform(min_values, max_values, (k, d))

        return cen
    except Exception:
        return None

def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset."""
    cen = initialize(X, k)

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if cen is None:
        return None, None

    n, d = X.shape

    labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - cen, axis=2), axis=1)

    for _ in range(iterations):
        old_cen = cen.copy()
        for i in range(len(cen)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                cen[i] = np.mean(cluster_points, axis=0)
            else:
                cen[i] = initialize(X, 1)
        
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - cen, axis=2), axis=1)
        
        if np.array_equal(old_cen, cen):
            break

    return cen, labels
