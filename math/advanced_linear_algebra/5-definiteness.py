#!/usr/bin/env python3
"""Imports"""
import numpy as np


def definiteness(matrix):
    """determines the definiteness of a given square matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)
    all_positive = np.all(eigenvalues > 0)
    all_nonnegative = np.all(eigenvalues >= 0)
    all_negative = np.all(eigenvalues < 0)
    all_nonpositive = np.all(eigenvalues <= 0)

    if all_positive:
        return "Positive definite"
    elif all_nonnegative:
        return "Positive semi-definite"
    elif all_negative:
        return "Negative definite"
    elif all_nonpositive:
        return "Negative semi-definite"
    else:
        return "Indefinite"
