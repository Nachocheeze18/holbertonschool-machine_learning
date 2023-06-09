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
        ph = (((h_prev - 1) * sh) + kh - h_prev) // 2 + 1
        pw = (((w_prev - 1) * sw) + kw - w_prev) // 2 + 1

    if padding == 'valid':
        ph = 0
        pw = 0

    A_prev = np.pad(A_prev, ((0, 0), (ph, ph),
                    (pw, pw), (0, 0)))

    dA_prev = np.pad(dA_prev, ((0, 0), (ph, ph),
                     (pw, pw), (0, 0)))

    for img in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    i = h * sh
                    j = w * sw
                    dW[:, :, :, c] += np.multiply(
                        A_prev[img, i:i + kh, j:j + kw, :],
                        dZ[img, h, w, c])
                    dA_prev[img, i:i + kh, j:j + kw, :] += (
                        np.multiply(W[:, :, :, c], dZ[img, h, w, c]))

    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph,
                          pw:-pw, :]

    return dA_prev, dW, db
