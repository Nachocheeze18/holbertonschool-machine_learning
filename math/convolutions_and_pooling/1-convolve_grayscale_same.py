#!/usr/bin/env python3
"""grayscale"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """By using this function, you can perform
      a same convolution on grayscale image
        with a specified kernel, producing
        transformed images that highlight
        certain features or extract useful
        information from the original images."""

    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    pad = np.pad(images, ((0, 0), (kh // 2, kh // 2),
                          (kw // 2,  kw // 2)), mode='constant')

    convolved_images = np.zeros((m, h, w))

    for j in range(h):
        for k in range(w):
            img = pad[:, j:j+kh, k:k+kw]
            convolved_images[:, j, k] = np.sum(img * kernel, axis=(1, 2))

    return convolved_images
