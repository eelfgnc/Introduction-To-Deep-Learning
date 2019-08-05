import matplotlib.pyplot as plt
from math import sqrt

n = 450
h = w = n/2;

a = [[0] * n for i in range(450)]
for i in range(n):
    for j in range(n):
        pixel = sqrt((i-w)**2 + (j-h)**2)
        if pixel<75:
            a[i][j] = [255, 0, 0] 
        elif 75<pixel<150:
            a[i][j] = [0, 0, 255]
        else:
            a[i][j] = [128, 128, 128]
            
imgplot = plt.imshow(a)
plt.show()