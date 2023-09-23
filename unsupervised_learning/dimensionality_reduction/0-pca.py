#!/usr/bin/env python3
"""Imports"""
import numpy as np


def pca(X, var=0.95):
    """Calculate the covariance matrix of the input data"""
    cov_matrix = np.cov(X, rowvar=False)

    values, vectors = np.linalg.eigh(cov_matrix)

    indices = np.argsort(values)[::-1]
    values = values[indices]
    vectors = vectors[:, indices]

    variance_ratio = np.cumsum(values) / np.sum(values)

    num = 3  # Compute the first three principal components

    W = vectors[:, :num]
    eigenvalues = values[:num]
    
    return W, eigenvalues
