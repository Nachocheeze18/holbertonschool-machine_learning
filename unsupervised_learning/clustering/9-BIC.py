#!/usr/bin/env python3
"""Imports"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization




def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ finds the best number of clusters for a GMM using
    the Bayesian Information Criterion"""
    if kmax is None:
        kmax = X.shape[0]

    best_k = None
    best_result = None
    best_bic = float('-inf')

    n, d = X.shape

    l = np.empty(kmax - kmin + 1)
    b = np.empty(kmax - kmin + 1)

    for k in range(kmin, kmax + 1):
        result = expectation_maximization(X, k, iterations, tol, verbose)

        if result is None:
            return None, None, None, None

        pi, _, S = result
        pi = np.array(pi)

        likelihood = result[3]

        num_params = k - 1 + k * d + k * d * (d + 1) // 2

        bic = k * np.log(n) - 2 * likelihood

        l[k - kmin] = likelihood
        b[k - kmin] = bic

        if bic > best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, result[1], S)

    return best_k, best_result, l, b
