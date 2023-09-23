#!/usr/bin/env python3
"""Imports"""
import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset."""
    _, S, Vt = np.linalg.svd(X, rowvar=False)
    total_variance = S**2 / np.sum(S**2)
    cumulative_variance = np.cumsum(total_variance)
    num_dimensions = np.argmax(cumulative_variance >= var) + 1
    W = Vt[:num_dimensions].T
    return W
