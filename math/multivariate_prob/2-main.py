#!/usr/bin/env python3

import numpy as np
from multinormal import MultiNormal

np.random.seed(6)
X = np.random.multivariate_normal([5, -4, 2], [[6, -3, 5], [-3, 10, -2], [5, -2, 5]], 10000).T
mn = MultiNormal(X)
x = X[:, 100:101]
print(mn.pdf(x))