import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

dataframe = pd.read_csv('Weather.csv')
# print(dataframe)

X = dataframe.values[:, 1].reshape(-1, 1)
y = dataframe.values[:, 0].reshape(-1, 1)

# plt.scatter(X, y, marker='o')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

def predict(new_radio, weight, bias):
    return weight*new_radio + bias

def cost_function(X, y, weight, bias):
    n = len(X)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] - (weight*X[i]+bias))**2
    return sum_error/n

def update_weight(X, y, weight, bias, learningRate):
    n = len(X)
    weight_temp = 0
    bias_temp = 0

    for i in range(n):
        weight_temp += -2*X[i]*(y[i]- (weight*X[i]+bias))
        bias_temp += -2*(y[i]- (weight*X[i]+bias))
    weight -= (weight_temp/n) * learningRate
    bias -= (bias/n) * learningRate
    return weight, bias

def train(X, y, weight, bias, learningRate, epochs):
    cost_hist= []
    for i in range(epochs):
        weight, bias = update_weight(X, y, weight, bias, learningRate)
        cost = cost_function(X, y, weight, bias)
        cost_hist.append(cost)
    return weight, bias, cost_hist

weight, bias, cost = train(X_train, y_train, 0.03, 0.0014, 0.00001, 1000)
print('KQ')
y_predictByManual = predict(X_test, weight,bias)

print('Mean Squared Error Sklearn: ', metrics.mean_squared_error(y_test, y_pred))
print('Mean Squared Error Gradient descent: ', metrics.mean_squared_error(y_test, y_predictByManual))

#show loss function
