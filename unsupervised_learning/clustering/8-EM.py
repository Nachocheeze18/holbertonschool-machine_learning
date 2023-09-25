#!/usr/bin/env python3
"""Imports"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs the expectation maximization for a GMM"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol <= 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    likelihood_s = 0

    for i in range(iterations + 1):
        g, likelihood_f = expectation(X, pi, m, S)

        if abs(likelihood_f - likelihood_s) <= tol:
            if verbose:
                likelihood_r = round(likelihood_f, 5)
                print("Log Likelihood after {} iterations: {:.5f}".format(
                    i, likelihood_r))
            return pi, m, S, g, likelihood_f

        if i < iterations:
            pi, m, S = maximization(X, g)

        if verbose and i % 10 == 0:
            likelihood_r = round(likelihood_f, 5)
            print("Log Likelihood after {} iterations: {:.5f}".format(
                i, likelihood_r))

        likelihood_s = likelihood_f

    return pi, m, S, g, likelihood_f
