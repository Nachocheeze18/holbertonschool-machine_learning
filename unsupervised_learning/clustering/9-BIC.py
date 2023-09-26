#!/usr/bin/env python3
"""Imports"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion."""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (type(kmax) is not int or kmax <= 0):
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if kmin >= kmax:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    n, d = X.shape
    values = []
    likelihoods = []
    results = []
    cluster_counts = []

    for k in range(kmin, kmax + 1):
        pi, m, S, _, likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)

        results.append((pi, m, S))
        cluster_counts.append(k)
        likelihoods.append(likelihood)

        p = k * d * (d + 1) / 2 + d * k + k - 1
        bic = p * np.log(n) - 2 * likelihood
        values.append(bic)

    values = np.array(values)

    best_index = np.argmin(values)

    return (cluster_counts[best_index], results[best_index],
            likelihoods, values)
