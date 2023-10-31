import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogitRegression:
    def __init__(self, learning_rate: int,  num_iterations: int):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.cost=[]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, y, y_pred):
        m = len(y)
        return (-1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        for i in range(n):
            self.theta[i] = np.random.uniform(0, 1)

        for v in range(self.num_iterations):
            z = np.dot(X, self.theta)
            y_pred = self.sigmoid(z)
            gradient = np.dot(X.T, (y_pred - y)) / m
            c = self.cost_function(y, y_pred)
            self.cost.append(c)
            self.theta -= self.learning_rate * gradient
        self.cost.reverse()

    def predict(self, X, y):
        z = np.dot(X, self.theta)
        y_pred = self.sigmoid(z)
        y_calc = []

        for i in y_pred:
            if i >= 0.5:
                y_calc.append(1)
            else:
                y_calc.append(0)
        res = 0
        for index in range(len(y_calc)):
            if y.iloc[index] == y_calc[index]:
                res += 1
        return res/len(y_calc)