#!/usr/bin/env python3
"""Imports"""
import numpy as np


def pca(X, ndim):
    """Perform PCA on a dataset with a specified number of dimensions."""
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:ndim].T
    T = X.dot(W)
    return T
