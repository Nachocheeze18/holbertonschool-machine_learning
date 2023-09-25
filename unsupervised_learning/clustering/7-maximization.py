#!/usr/bin/env python3
"""Imports"""
import numpy as np


def maximization(X, g):
    """calculates the maximization step in the EM algorithm for a GMM"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if False in np.isclose(g.sum(axis=0), np.ones((g.shape[1]))):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    pi = np.zeros(k)
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    if g.shape[1] != n:
        return None, None, None

    pi = np.sum(g, axis=1) / n

    for i in range(k):
        weighted_data = g[i].reshape(-1, 1) * X
        m[i] = np.sum(weighted_data, axis=0) / np.sum(g[i])
        
        diff = X - m[i]
        S[i] = np.dot(g[i].reshape(1, -1) * diff.T, diff) / np.sum(g[i])

    return pi, m, S
