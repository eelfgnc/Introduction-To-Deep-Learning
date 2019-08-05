import numpy as np

def nnloss(x, t, z):
    instanceWeights = np.ones(x.shape)
    res = np.subtract(x, t)
    if z == 0:
        y = np.dot((1/2) * instanceWeights[:].T , np.power(res, 2))
    else:
        y = res        
    return y