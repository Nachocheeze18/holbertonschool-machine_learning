#!/usr/bin/env python3
"""Imports"""
import numpy as np


def pca(X, ndim):
    """Perform PCA on a dataset with a specified number of dimensions."""
    mean = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(mean, full_matrices=False)
    W = Vt[:ndim].T
    T = np.dot(mean, W)
    return T
