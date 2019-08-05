import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from skimage import color
from PIL import Image
import random
from sklearn.utils import shuffle

#activation function
def sigmoid_function(sum): 
    return 1 / (1 + np.exp(-sum))

def sigmoid_derivation(sum):
    return sigmoid_function(sum)*(1 - sigmoid_function(sum))

def trainPerceptron(inputs, t, weights, rho, iterNo):
    for i in range(iterNo):
        for j in range(3):
            a = inputs[j]
            sum_wa = np.dot(weights[0],np.transpose(a))
            y = sigmoid_function(sum_wa)
            y_der = sigmoid_derivation(y)
            target = t[j][0]
            error = target - y
            delta_w = np.dot(rho * y_der * error, inputs[j])
            weights[j] = delta_w + weights[j]
    return weights

def testPerceptron(test_inputs, weights):
    sum_test = np.dot(weights,np.transpose(test_inputs))
    y = sigmoid_function(sum_test)
    return  y

train = [[]*3072]*18
bias = np.ones((18,1))
t = np.ones((18,1))
temp = [3072]
j=0
for i in range(1, 10):
    imgmobile = misc.imread('train/automobile/automobile' + (i).__str__() +'.png')
    imgdog = misc.imread('train/dog/dog' + (i).__str__() +'.png')
    mobile = imgmobile.flatten()
    dog = imgdog.flatten()
    train[j] = mobile
    t[j] = 0
    j = j + 1
    train[j] = dog
    t[j] = 1
    j = j + 1
    
train = np.hstack((train,bias))
train, t = shuffle(train, t)
test = [0.001]
weights = test * 3073
w = trainPerceptron(train, t, weights , 0.01, 100)

imgmobile =  misc.imread('test/automobile/automobile10.png')
test_inputs = imgmobile.flatten()
bias = [1]
test_inputs = np.hstack((test_inputs,bias))
sonuc = testPerceptron(test_inputs, w)
print('automobile(0): ', sonuc)

imgdog =  misc.imread('test/dog/dog10.png')
test_inputs = imgdog.flatten()
bias = [1]
test_inputs = np.hstack((test_inputs,bias))
sonuc = testPerceptron(test_inputs, w)
print('dog(1): ', sonuc)
