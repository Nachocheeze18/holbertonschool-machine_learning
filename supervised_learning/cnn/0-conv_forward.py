#!/usr/bin/env python3
"""forward prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """performs forward propagation over a
    convolutional layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    else:  # padding == "valid"
        ph, pw = 0, 0

    h_new = int((h_prev + 2 * ph - kh) / sh) + 1
    w_new = int((w_prev + 2 * pw - kw) / sw) + 1

    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph),
                                    (pw, pw), (0, 0)), mode='constant')
    Z = np.zeros((m, h_new, w_new, c_new))

    for x in range(h_new):
        for y in range(w_new):
            for z in range(c_new):
                i_start, i_end = x * sh, x * sh + kh
                j_start, j_end = y * sw, y * sw + kw
                A_slice = A_prev_padded[:, i_start:i_end, j_start:j_end, :]
                Z[:, x, y, z] = np.sum(A_slice * W[:, :, :, z],
                                       axis=(1, 2, 3)) + b[:, :, :, z]

    A = activation(Z)
    return A
