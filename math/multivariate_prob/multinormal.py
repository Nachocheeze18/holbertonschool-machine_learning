#!/usr/bin/env python3
"""imports"""
import numpy as np


class MultiNormal:
    """multinormal class"""
    def __init__(self, data):
        """class constructor"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        n, d = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data)
        self.variance = np.var(data)

    def pdf(self, x):
        """calculates the Probability Density Function"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        n, d = x.shape  # Get the number of data points and dimensionality

        if d != 1:
            raise ValueError("Each data point in x must have dimensionality 1")

        exponent = -0.5 * ((x - self.mean) ** 2) / self.variance
        denominator = np.sqrt(2 * np.pi * self.variance)
        value = np.exp(exponent) / denominator
        return value