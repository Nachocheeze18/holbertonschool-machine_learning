#!/usr/bin/env python3
"""backward pool"""

import numpy as np

def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs back propagation over a pooling layer of a neural network"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for img in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    if mode == 'max':
                        mask = (A_prev[img, v_start:v_end, h_start:h_end, c] == np.max(A_prev[img, v_start:v_end, h_start:h_end, c]))
                        dA_prev[img, v_start:v_end, h_start:h_end, c] += mask * dA[img, h, w, c]
                    elif mode == 'avg':
                        avg = dA[img, h, w, c] / (kh * kw)
                        dA_prev[img, v_start:v_end, h_start:h_end, c] += np.ones((kh, kw)) * avg

    return dA_prev
