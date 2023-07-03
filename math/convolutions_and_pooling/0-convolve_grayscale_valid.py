#!/usr/bin/env python3
"""convolve"""

import numpy as np

def convolve_grayscale_valid(images, kernel):
    """function that performs a valid convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    convolved_images = np.zeros((m, h - kh + 1, w - kw + 1))
    
    for i in range(m):
        for j in range(h - kh + 1):
            for k in range(w - kw + 1):
                convolved_images[i, j, k] = np.sum(images[i, j:j+kh, k:k+kw] * kernel)
    
    return convolved_images