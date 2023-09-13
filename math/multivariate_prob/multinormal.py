#!/usr/bin/env python3
"""imports"""
import numpy as np


class MultiNormal:
    """Multivariate Normal distribution class"""
    def __init__(self, data):
        """Class constructor"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        
        n, d = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate the mean of the data
        self.mean = np.mean(data, axis=1).reshape(-1, 1)

        # Calculate the covariance matrix using the unbiased formula
        mean_centered_data = data - self.mean
        self.cov = (1 / (n - 1)) * (mean_centered_data @ mean_centered_data.T)

    def pdf(self, x):
        """calculates the Probability Density Function"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.mean.shape[0], 1):
            raise ValueError("x must have the shape ({}, 1)".format(self.mean.shape[0]))

        d = self.mean.shape[0]

        exponent = -0.5 * ((x - self.mean).T @ np.linalg.inv(self.cov) @ (x - self.mean))[0, 0]
        denominator = (2 * np.pi) ** (-d / 2) * np.sqrt(np.linalg.det(self.cov))
        value = np.exp(exponent) / denominator
        return value