#!/usr/bin/env python3
"""back prop"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ performs back propagation over a convolutional
    layer of a neural network"""
    m = dZ.shape[0]
    h_new = dZ.shape[1]
    w_new = dZ.shape[2]

    c_new = b.shape[3]

    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = W.shape[0]
    kw = W.shape[1]

    sh = stride[0]
    sw = stride[1]

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        pad_top_bottom = (((h_prev - 1) * sh) + kh - h_prev) // 2 + 1
        pad_left_right = (((w_prev - 1) * sw) + kw - w_prev) // 2 + 1

    if padding == 'valid':
        pad_top_bottom = 0
        pad_left_right = 0

    A_prev = np.pad(A_prev, ((0, 0), (pad_top_bottom, pad_top_bottom),
                    (pad_left_right, pad_left_right), (0, 0)))

    dA_prev = np.pad(dA_prev, ((0, 0), (pad_top_bottom, pad_top_bottom),
                     (pad_left_right, pad_left_right), (0, 0)))

    for image in range(m):
        for x in range(h_new):
            for y in range(w_new):
                for z in range(c_new):
                    i = x * sh
                    j = y * sw
                    dW[:, :, :, z] += np.multiply(
                        A_prev[image, i:i + kh, j:j + kw, :],
                        dZ[image, x, y, z])
                    dA_prev[image, i:i + kh, j:j + kw, :] += (
                        np.multiply(W[:, :, :, z], dZ[image, x, y, z]))

    if padding == 'same':
        dA_prev = dA_prev[:, pad_top_bottom:-pad_top_bottom,
                          pad_left_right:-pad_left_right, :]

    return dA_prev, dW, db
