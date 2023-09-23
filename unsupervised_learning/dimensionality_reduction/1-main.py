#!/usr/bin/env python3

import numpy as np
pca = __import__('0-pca').pca

import_data = np.load('data.npz')

np.random.seed(0)
a = import_data['a']
b = import_data['b']
c = import_data['c']
d = 2 * a
e = -5 * b
f = 10 * c

X = np.array([a, b, c, d, e, f]).T
m = X.shape[0]

X_m = X - np.mean(X, axis=0)
W = pca(X_m)
print(W)
print(W.shape)
W = pca(X_m, var=0.59)
print(W)
print(W.shape)