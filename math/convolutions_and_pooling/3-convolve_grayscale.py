#!/usr/bin/env python3
"""convolution on grayscale images"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """his function allows you to perform convolutions on grayscale images
    using a specified kernel, padding option, and stride values. The
    convolved images are returned as a NumPy array."""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        ph = (((h - 1) * sh) + kh - h) // 2 + 1
        pw = (((w - 1) * sw) + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph = padding[0]
        pw = padding[1]

    pad = np.pad(images, ((0, 0), (ph, ph),
                          (pw,  pw)))

    h = (h + 2 * ph - kh) // sh + 1
    w = (w + 2 * pw - kw) // sw + 1

    convolved_images = np.zeros((m, h, w))

    for j in range(h):
        for k in range(w):
            y = j * sh
            z = k * sw
            img = np.multiply(pad[:, y:y+kh, z:z+kw], kernel)
            convolved_images[:, j, k] = np.sum(img, axis=(1, 2))

    return convolved_images
