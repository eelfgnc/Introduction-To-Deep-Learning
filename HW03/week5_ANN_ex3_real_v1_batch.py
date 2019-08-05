import numpy as np
import pandas as pd
import getIrısData as data
import sigm
import matplotlib.pyplot as plt
import nnloss

grap = np.zeros((2, 10))
l=0

trainingSet, testSet, y1, y2 = data.getIrısData()

batchSize = 5
input = trainingSet
groundTruth = y1
coeff = 0.1
iteration = 1000
inputLength = input.shape[0]
hiddenN = 5
outputN = groundTruth.shape[1]
num_layers = 3
tol = 0.1

stack1_w = np.random.rand(hiddenN, inputLength)
stack1_b = np.random.rand(hiddenN, 1)
stack2_w = np.random.rand(outputN, hiddenN)
stack2_b = np.random.rand(outputN, 1)

outputStack1 = np.zeros((5, 1))
outputStack2 = np.zeros((5, 1))
outputStack3 = np.zeros((3, 1))

gradStack_epsilon1 = np.zeros((5, 1))
gradStack_epsilon2 = np.zeros((3, 1))

inputs = np.zeros((5, 1))
p = np.zeros((3, 1))
for i in range(iteration):
    err = 0
    border = y1.shape[0]
    for j in range(0, border, batchSize):
        data = input[:, j: j+batchSize]
        labels = groundTruth[j: j+batchSize, :]
        cost = 0
        for kk in range(batchSize):
            inputs[0:5, 0] = data[:, kk]
            outputStack1 = inputs
            
            #forward propagation
            outputStack2 = np.subtract(np.dot(stack1_w, outputStack1), stack1_b)
            outputStack2 = sigm.sigm(outputStack2)
            outputStack3 = np.subtract(np.dot(stack2_w, outputStack2), stack2_b)
            outputStack3 = sigm.sigm(outputStack3)
            
            p[:, 0] = outputStack3[:, 0]
            epsilon = nnloss.nnloss(labels[kk: kk+1, :].T, p, 1)
            cost = cost + nnloss.nnloss(labels[kk: kk+1, :].T, p, 0)
            err = err + cost
            if j == 0:
                gradStack_epsilon2 = np.multiply(np.multiply(outputStack3, 1-outputStack3), epsilon)
                epsilon = np.dot(stack2_w.T, gradStack_epsilon2)
                gradStack_epsilon1 = np.multiply(np.multiply(outputStack2, 1-outputStack2), epsilon)
                epsilon = np.dot(stack1_w.T, gradStack_epsilon1)
            else:
                gradStack_epsilon2 += np.multiply(np.multiply(outputStack3, 1-outputStack3), epsilon)
                epsilon = np.dot(stack2_w.T, gradStack_epsilon2)
                gradStack_epsilon1 += np.multiply(np.multiply(outputStack2, 1-outputStack2), epsilon)
                epsilon = np.dot(stack1_w.T, gradStack_epsilon1)
                
        gradStack_epsilon2 = gradStack_epsilon2 / batchSize
        gradStack_epsilon1 = gradStack_epsilon1 / batchSize
        cost = cost/batchSize
        err = err + cost
        
        stack1_w = stack1_w + np.multiply(coeff, np.dot(gradStack_epsilon1, outputStack1.T))
        stack1_b = stack1_b + np.multiply((coeff*(-1)), gradStack_epsilon1)
        stack2_w = stack2_w + np.multiply(coeff, np.dot(gradStack_epsilon2, outputStack2.T))
        stack2_b = stack2_b + np.multiply((coeff*(-1)), gradStack_epsilon2)
    
    if (i%100) == 0:
        grap[0, l] = i
        grap[1, l] = err
        l = l + 1
    if abs(err)<tol:
        break

plt.plot(grap[0, 0: 10], grap[1, 0: 10], 'b*-')
plt.axis([0, 1000, 0, 100])
plt.show()

np.save('stack1_b', stack1_b)
np.save('stack1_w', stack1_w)
np.save('stack2_b', stack2_b)
np.save('stack2_w', stack2_w)

testSet = np.load('testSet.npy')
y2 = np.load('y2.npy')

#Test the code
input = testSet
tol=0.1
groundTruth = y2
out = np.zeros([y2.shape[0],y2.shape[1]])
count = 0

for j in range(y2.shape[0]):
    inputs[:, 0] = input[:, j]
    outputStack1 = inputs
    
    #forward propagation
    outputStack2 = np.subtract(np.dot(stack1_w, outputStack1), stack1_b)
    outputStack2 = sigm.sigm(outputStack2)
    
    outputStack3 = np.subtract(np.dot(stack2_w, outputStack2), stack2_b)
    outputStack3 = sigm.sigm(outputStack3)
    
    out[j, :] = outputStack3.T
    truth = groundTruth[j, :]
    o = out[j, :]
    epsilon = np.subtract(truth, o)
    err = np.sum(np.power(epsilon, 2))
    if err<tol:
        count=count+1

acc = (count/(out.shape[0])) * 100
print('accuracy of system: ', acc)