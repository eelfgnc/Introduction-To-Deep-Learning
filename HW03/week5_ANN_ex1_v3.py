import numpy as np
import pandas as pd
import getIrısData as data
import sigm
import matplotlib.pyplot as plt
import nnloss
import sigmDerivative

input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
input = input.T
groundTruth = np.array([[0], [1], [1], [0]])

bias = np.zeros((1, 3))
bias[0, 0:3] = [-1, -1, -1]

coeff = 0.7
iteration = 10000
inputLength = 2
hiddenN = 2
outputN = 1

w1 = np.ones((hiddenN, hiddenN + 1))
w1[0, 0:3] = 0.1
w1[1, 0:3] = 0.2

w2 = np.ones((outputN, hiddenN + 1))
w2[0, 0:3] = 0.3

inputs = np.ones((2, 1))
hl1 = np.ones((2, 1))
HiddenLayerOutput1 = np.ones((2, 1))
h = np.ones((3,1))
k = np.ones((3, 1))
c = np.ones((2, 1))
d = np.ones((2, 1))
e = np.ones((1, 3))
f = np.ones((1, 3))

for i in range(iteration):
    out = np.zeros((4, 1))
    for j in range(4):
        inputs[:, 0] = input[:, j]
        k[0, 0] = -1
        k[1, 0] = inputs[0, 0]
        k[2, 0] = inputs[1, 0]
        
        #Hidden Layer 1
        hl1 = np.dot(w1, k)
        HiddenLayerOutput1 = sigm.sigm(hl1)
        
        #Output Layer
        h[0, 0] = -1
        h[1, 0] = HiddenLayerOutput1[0, 0]
        h[2, 0] = HiddenLayerOutput1[1, 0]
        
        #Output Layer
        x3_1 = np.dot(w2, h)
        out[j, 0] = sigm.sigm(x3_1)
        delta3 = np.multiply(sigmDerivative.sigmDerivative(x3_1), groundTruth[j,0] - out[j,0])
        
        c = sigmDerivative.sigmDerivative(hl1)
        d[0,0] = w2[0,1]
        d[1,0] = w2[0,2]
        delta2 = np.multiply(np.multiply(c,d), delta3)
        
        e[0, 0] = -1
        e[0, 1] = HiddenLayerOutput1[0, 0]
        e[0, 2] = HiddenLayerOutput1[1, 0]
        
        w2 = w2 + np.multiply(np.multiply(coeff,e), delta3)
        
        f[0, 0] = -1
        f[0, 1] = inputs[0, 0]
        f[0, 2] = inputs[1, 0]
        
        w1 = w1 + np.multiply(coeff, np.dot(delta2, f))

#Test Code
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
input = input.T
groundTruth = np.array([[0], [1], [1], [0]])
out = np.zeros((4, 1))

for j in range(4):
    inputs[:, 0] = input[:, j]
    
    #Hidden Layer
    k[0, 0] = -1
    k[1, 0] = inputs[0, 0]
    k[2, 0] = inputs[1, 0] 
    hl1 = np.dot(w1, k)
    
    HiddenLayerOutput1 = sigm.sigm(hl1)
    
    #Output Layer
    h[0, 0] = -1
    h[1, 0] = HiddenLayerOutput1[0, 0]
    h[2, 0] = HiddenLayerOutput1[1, 0]
    x3_1 = np.dot(w2, h)
    
    out[j,0] = sigm.sigm(x3_1)























