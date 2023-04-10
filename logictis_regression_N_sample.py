#Logictis regression N sample

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def train(X, y, learning_rate=0.01, epochs=1000):

    theta = np.random.random((X.shape[1]+1, 1))
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    losses = []

    for epoch in range(epochs):
        y_pred = sigmoid(np.dot(X, theta))
        #loss binary cross entropy
        loss = -y*np.log(y_pred) - (1-y)*np.log(1-y_pred)
        losses.append(np.mean(loss))
        #gradient
        gradient = (np.dot(X.T, (y_pred-y))) / len(X)
        #update theta
        theta -= learning_rate*gradient

    return theta, losses
