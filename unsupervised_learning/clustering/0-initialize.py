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
