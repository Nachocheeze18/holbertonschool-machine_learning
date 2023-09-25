#!/usr/bin/env python3
"""Imports"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    if X.shape[1] != m.shape[1] or m.shape[1] != S.shape[1]:
        return None, None
    if S.shape[1] != S.shape[2]:
        return None, None
    if pi.shape[0] != m.shape[0] or m.shape[0] != S.shape[0]:
        return None, None
    if False in [np.isclose(pi.sum(), 1)]:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    g = np.zeros((k, n))

    for i in range(k):
        P = pi[i] * pdf(X, m[i], S[i])
        g[i] = P

    g_sum = np.sum(g, axis=0)
    g /= g_sum

    likelihood = np.sum(np.log(g_sum))

    return g, likelihood
