#!/usr/bin/python3

import numpy as np
from sklearn import linear_model

def PredictFunction(theta, x):
  if not isinstance(theta, np.ndarray):
    raise TypeError('theta: numpy.ndarray')
  if not isinstance(x, np.ndarray):
    raise TypeError('x: numpy.ndarray')

  if len(x.shape) != 2:
    raise ValueError('x: a matrix/vector')
  # x: a matrix having m rows x n columns
  m = x.shape[0]
  n = x.shape[1]

  # theta: a column vector having n + 1 rows
  if (len(theta.shape) != 2) or (theta.shape[0] != n + 1) or (theta.shape[1] != 1):
    raise ValueError('theta: a column vector having {} rows'.format(n + 1))

  new_x = np.concatenate((np.ones((m, 1)), x), axis = 1)
  # ====================== YOUR CODE HERE ======================
  result = new_x @ theta
  # ============================================================
  return result

def LossFunction(theta, X, y):
  if not isinstance(theta, np.ndarray):
    raise TypeError('theta: numpy.ndarray')
  if not isinstance(X, np.ndarray):
    raise TypeError('X: numpy.ndarray')
  if not isinstance(y, np.ndarray):
    raise TypeError('y: numpy.ndarray')

  if len(X.shape) != 2:
    raise ValueError('x: a matrix/vector')
  # X: a matrix having m rows x n columns
  m = X.shape[0]
  n = X.shape[1]

  # theta: a column vector having n + 1 rows
  if (len(theta.shape) != 2) or (theta.shape[0] != n + 1) or (theta.shape[1] != 1):
    raise ValueError('theta: a column vector having {} rows'.format(n + 1))

  # y: a column vector having m rows
  if (len(y.shape) != 2) or (y.shape[0] != m) or (y.shape[1] != 1):
    raise ValueError('y: a column vector having {} rows'.format(m))

  new_X = np.concatenate((np.ones((m, 1)), X), axis = 1)
  # ====================== YOUR CODE HERE ======================
  result = np.sum(np.power(new_X @ theta - y, 2))
  # ============================================================
  return result

def GradientDescent(L, X, y):
  if not isinstance(X, np.ndarray):
    raise TypeError('X: numpy.ndarray')
  if not isinstance(y, np.ndarray):
    raise TypeError('y: numpy.ndarray')
  if L <= 0:
    raise ValueError('alpha > 0')

  if len(X.shape) != 2:
    raise ValueError('x: a matrix/vector')
  # X: a matrix having m rows x n columns
  # m = X.shape[0]
  # n = X.shape[1]

  # y: a column vector having m rows
  # if (len(y.shape) != 2) or (y.shape[0] != m) or (y.shape[1] != 1):
  #   raise ValueError('y: a column vector having {} rows'.format(m))

  # theta = np.random.rand(n + 1, 1)
  # new_X = np.concatenate((np.ones((m, 1)), X), axis = 1)

  m = 0
  c = 0
  n = float(len(X))
  # For simplicity, set iterations = 1500
  epochs = 1500
  for i in range(epochs):
    # ====================== YOUR CODE HERE ======================
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    # ============================================================
  print (Y_pred)
  
  return ()
  
def CalculateDirectly(X, y):
  if not isinstance(X, np.ndarray):
    raise TypeError('X: numpy.ndarray')
  if not isinstance(y, np.ndarray):
    raise TypeError('y: numpy.ndarray')

  if len(X.shape) != 2:
    raise ValueError('x: a matrix/vector')
  # X: a matrix having m rows x n columns
  m = X.shape[0]
  n = X.shape[1]

  # y: a column vector having m rows
  if (len(y.shape) != 2) or (y.shape[0] != m) or (y.shape[1] != 1):
    raise ValueError('y: a column vector having {} rows'.format(m))

  new_X = np.concatenate((np.ones((m, 1)), X), axis = 1)
  # ====================== YOUR CODE HERE ======================
  theta = np.linalg.inv(new_X.T @ new_X) @ new_X.T @ y
  # ============================================================
  return theta

def LinearRegression(X, y):
  if not isinstance(X, np.ndarray):
    raise TypeError('X: numpy.ndarray')
  if not isinstance(y, np.ndarray):
    raise TypeError('y: numpy.ndarray')

  if len(X.shape) != 2:
    raise ValueError('x: a matrix/vector')
  # X: a matrix having m rows x n columns
  m = X.shape[0]
  n = X.shape[1]

  # y: a column vector having m rows
  if (len(y.shape) != 2) or (y.shape[0] != m) or (y.shape[1] != 1):
    raise ValueError('y: a column vector having {} rows'.format(m))

  new_X = np.concatenate((np.ones((m, 1)), X), axis = 1)
  # ====================== YOUR CODE HERE ======================
  theta = linear_model.LinearRegression(fit_intercept=False). fit(new_X, y).coef_.T
  # ============================================================
  return theta
