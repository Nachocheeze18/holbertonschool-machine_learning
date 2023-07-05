#!/usr/bin/env python3
"""convolve"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """function that performs a valid convolution on grayscale images"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    image = np.zeros((m, h - kh + 1, w - kw + 1))

    for j in range(h - kh + 1):
        for k in range(w - kw + 1):
            i = np.multiply(images[:, j:j + kh, k:k + kw], kernel)
            image[:, j, k] = np.sum(i, axis=(1, 2))

    return image
