#!/usr/bin/env python3
"""Imports"""
import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset."""
    cov_matrix = np.cov(X, rowvar=False)

    U, S, Vt = np.linalg.svd(cov_matrix)

    variance = np.sum(S)

    cumulative_variance = np.cumsum(S) / variance
    num = np.argmax(cumulative_variance >= var) + 1

    W = Vt[:num, :].T

    return W
