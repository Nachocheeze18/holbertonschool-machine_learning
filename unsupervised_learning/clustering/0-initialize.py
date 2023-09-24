#!/usr/bin/env python3
"""Imports"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means"""
    if k <= 0 or k > X.shape[0]:
        return None

    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    centroids = np.random.uniform(min_values, max_values, size=(k, X.shape[1]))

    return centroids
