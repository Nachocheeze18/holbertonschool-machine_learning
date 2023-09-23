#!/usr/bin/env python3
"""Imports"""
import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset."""
    cov_matrix = np.cov(X, rowvar=False)
    _, S, Vt = np.linalg.svd(cov_matrix)
    total_variance = np.sum(S)
    cumulative_variance = np.cumsum(S) / total_variance
    num_dimensions_to_keep = np.argmax(cumulative_variance >= var) + 1
    W = Vt.T[:, :num_dimensions_to_keep]
    return W
