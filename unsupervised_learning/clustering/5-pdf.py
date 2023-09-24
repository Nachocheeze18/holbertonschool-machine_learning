#!/usr/bin/env python3
"""Imports"""
import numpy as np


def pdf(X, m, S):
    """calculates the probability density function
    of a Gaussian distribution"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None

    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    if X.shape[1] != m.shape[0]:
        return None

    if S.shape[0] != S.shape[1]:
        return None

    if m.shape[0] != S.shape[0]:
        return None

    try:
        _, d = X.shape
        inv_S = np.linalg.inv(S)
        det_S = np.linalg.det(S)

        if det_S <= 0:
            return None

        X_minus_m = X - m
        exponent = -0.5 * np.sum(X_minus_m @ inv_S * X_minus_m, axis=1)
        coeff = 1 / np.sqrt((2 * np.pi) ** d * det_S)

        pdf_values = coeff * np.exp(exponent)
        pdf_values = np.maximum(pdf_values, 1e-300)

        return pdf_values

    except np.linalg.LinAlgError:
        return None
