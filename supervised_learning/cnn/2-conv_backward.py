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

    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = W.shape[0]
    kw = W.shape[1]
    c_prev = W.shape[2]
    c_new = W.shape[3]

    sh = stride[0]
    sw = stride[1]

    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    elif padding == "valid":
        ph = 0
        pw = 0

    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

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