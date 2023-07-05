#!/usr/bin/env python3
"""stride tuples"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    kh = kernel.shape[0]
    kw = kernel.shape[1]
    kc = kernel.shape[2]

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
            convolved_images[:, j, k] = np.sum(img, axis=(1, 2, 3))

    return convolved_images
