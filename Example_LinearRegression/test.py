#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ex_01 import *

# Read data 'ex1data1.txt'
data = pd.read_csv('data.txt', header = None)
data = np.array(data)
m = data.shape[0]
n = data.shape[1] - 1
X = np.reshape(data[:, 0], (m, n)) #independent value
y = np.reshape(data[:, 1], (m, 1))
print(y)

# Plot data
plt.plot(X, y, 'rx')
plt.show()

# Linear Regression
print('='*80)
theta = np.zeros((2, 1))
print(theta)
print('Initial loss = {}'.format(LossFunction(theta, X, y)))

alpha = 0.0001
theta_gradient_descent = GradientDescent(alpha, X, y)
theta_calculate_directly = CalculateDirectly(X, y)
theta_linear_regression = LinearRegression(X, y)

print('='*80)
print('Plot results ... ')
plt.plot(X, y, 'rx')
plt.plot([np.min(X), np.max(X)], np.array([[1, np.min(X)], [1, np.max(X)]]) @ theta_gradient_descent, ls = '-')
plt.plot([np.min(X), np.max(X)], np.array([[1, np.min(X)], [1, np.max(X)]]) @ theta_calculate_directly, ls = '--')
plt.plot([np.min(X), np.max(X)], np.array([[1, np.min(X)], [1, np.max(X)]]) @ theta_linear_regression, ls = '-.')
plt.show()

print('='*80)
new_X = np.array([[1.8], [2.5]])
print('X =\n{}'.format(new_X))
print('With gradient descent method, we predict y =\n{}'.format(PredictFunction(theta_gradient_descent, new_X)))
print('With calculate directly method, we predict y =\n{}'.format(PredictFunction(theta_calculate_directly, new_X)))
print('With linear regression method, we predict y =\n{}'.format(PredictFunction(theta_linear_regression, new_X)))
