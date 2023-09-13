#!/usr/bin/env python3

import numpy as np
from multinormal import MultiNormal

np.random.seed(2)
X = np.random.multivariate_normal([5, -4, 2], [[6, -3, 5], [-3, 10, -2], [5, -2, 5]], 10000).T
mn = MultiNormal(X)
print(mn.mean)
with open("3-test", "w+") as f:
    f.write(str(mn.cov))