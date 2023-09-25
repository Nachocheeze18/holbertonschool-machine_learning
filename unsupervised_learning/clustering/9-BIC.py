#!/usr/bin/env python3
"""Imports"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None:
        if not isinstance(kmax, int) or kmax <= 0 or kmin >= kmax:
            return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (float, int)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    best_bic = float('-inf')
    best_k = None
    best_result = None
    likelihoods = []
    bic_values = []

    for k in range(kmin, kmax + 1):
        result = expectation_maximization(X, k, iterations, tol, verbose)

        if result is None:
            return None, None, None, None

        pi, m, S, _, likelihood = result
        likelihoods.append(likelihood)

        num_params = k + k * d + k * d * (d + 1) // 2 

        bic = num_params * np.log(n) - 2 * likelihood
        bic_values.append(bic)

        if bic > best_bic:
            best_bic = bic
            best_k = k
            best_result = result

    return best_k, best_result, np.array(likelihoods), np.array(bic_values)
