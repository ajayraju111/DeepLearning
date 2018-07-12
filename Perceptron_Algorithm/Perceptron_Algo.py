# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:21:26 2018

@author: Rishabh Sharma
"""

import numpy as np
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])


def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        pred = prediction(X[i],W,b)
        if y[i]-pred == 1:
            for j in range(len(W)):
                W[j]+=X[i][j]*learn_rate
            b += learn_rate
        elif y[i]-pred == -1:
            for j in range(len(W)):
                W[j]-=X[i][j]*learn_rate
            b -= learn_rate
    return W,b
    
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    
    boundary_lines = []
    for i in range(num_epochs):
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
