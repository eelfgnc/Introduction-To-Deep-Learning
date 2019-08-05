import math
import numpy as np
def sigm(X):
    row = X.shape[0]
    col = X.shape[1]
    x = np.zeros((row, col))
    for i in range(row):
        y = X[i, 0]
        y = (-1)*y
        z = math.exp(y)
        x[i, 0] = (1 / (1 + z))
    return x