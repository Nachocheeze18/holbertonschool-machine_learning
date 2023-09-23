#!/usr/bin/env python3
"""Imports"""
import numpy as np


def pca(X, ndim):
    """Perform PCA on a dataset with a specified number of dimensions."""
    mean = np.mean(X, axis=0)
    centered_data = X - mean
    cov_matrix = np.cov(centered_data, rowvar=False)
    values, vectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(values)[::-1]
    values = values[sorted_indices]
    vectors = vectors[:, sorted_indices]
    top_vectors = vectors[:, :ndim]
    T = np.dot(centered_data, top_vectors)
    return T
