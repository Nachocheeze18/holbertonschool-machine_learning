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

np.random.seed(0)
m = np.random.randint(1000, 2000)
h, w = np.random.randint(100, 200, 2).tolist()
fh, fw = np.random.randint(3, 10, 2).tolist()

images = np.random.randint(0, 256, (m, h, w))
kernel = np.random.randint(0, 10, (fh, fw))
conv_ims = convolve_grayscale_valid(images, kernel)
print(conv_ims)
print(conv_ims.shape)
