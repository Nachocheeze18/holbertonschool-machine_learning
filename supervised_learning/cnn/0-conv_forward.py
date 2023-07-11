import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = int(np.ceil((h_prev * sh - sh + kh - h_prev) / 2))
        pad_w = int(np.ceil((w_prev * sw - sw + kw - w_prev) / 2))
    else:
        pad_h, pad_w = 0, 0

    h_out = int((h_prev - kh + 2 * pad_h) / sh) + 1
    w_out = int((w_prev - kw + 2 * pad_w) / sw) + 1

    if pad_h > 0 or pad_w > 0:
        A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    else:
        A_prev_pad = A_prev

    Z = np.zeros((m, h_out, w_out, c_new))

    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                v_start = h * sh
                v_end = v_start + kh
                h_start = w * sw
                h_end = h_start + kw
                A_slice = A_prev_pad[i, v_start:v_end, h_start:h_end, :]

                Z[i, h, w, :] = np.sum(A_slice * W, axis=(0, 1, 2)) + b


    A = activation(Z)

    return A