#!/usr/bin/env python3
"""padding"""
import numpy as np


def convolve_grayscale_same(images, kernel, padding):
    """By using this function, you can perform
    a convolution on grayscale images with a
    specified kernel and custom padding,
    obtaining transformed images that
    incorporate the padding to preserve
    the image dimensions during convolution.."""

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    ph = padding[0]
    pw = padding[1]

    m = images.shape[0]
    h = images.shape[1] + (2 * padding[0]) - kh + 1
    w = images.shape[2] + (2 * padding[1]) - kw + 1

    pad = np.pad(images, ((0, 0), (ph, ph),
                          (pw,  pw)), mode='constant')

    convolved_images = np.zeros((m, h, w))

    for j in range(h):
        for k in range(w):
            img = pad[:, j:j+kh, k:k+kw]
            convolved_images[:, j, k] = np.sum(img * kernel, axis=(1, 2))

    return convolved_images
