#!/usr/bin/env python3
"""Forward Prop """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs forward propagation over a pooling layer of a neural network"""
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    output_new = np.zeros((m, h_new, w_new, c_prev))

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_prev):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    input_slice = A_prev[i, h_start:h_end, w_start:w_end, c]

                    if mode == 'max':
                        output_new[i, h, w, c] = np.max(input_slice)
                    elif mode == 'avg':
                        output_new[i, h, w, c] = np.mean(input_slice)

    return output_new
