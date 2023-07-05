#!/usr/bin/env python3
"""performs pooling on images"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """function allows you to perform either max pooling
    or average pooling on the input images based on the
    specified kernel shape, stride, and mode."""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    
    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    pad = np.zeros((m, (h - kh) // sh + 1, (w - kw) // sw + 1, c))

    for j in range((h - kh) // sh + 1):
        for k in range((w - kw) // sw + 1):
            img = images[:, j*sh:j*sh+kh, k*sw:k*sw+kw, :]

            if mode == 'max':
                pad[:, j, k, :] = np.max(img, axis=(1, 2))
            elif mode == 'avg':
                pad[:, j, k, :] = np.mean(img, axis=(1, 2))

    return pad