import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# %matplotlib inline

# Summary data

dataset = pd.read_csv('Weather.csv', low_memory=False)
print(dataset.shape)
print(dataset.describe())

# chart

# dataset.plot(x='MinTemp', y='MaxTemp', style='o')
# plt.title('MinTemp vs MaxTemp')
# plt.xlabel('MinTemp')
# plt.ylabel('MaxTemp')

# plt.figure(figsize=(15, 10))
# plt.tight_layout()
# seabornInstance.distplot(dataset['MaxTemp'])
# plt.show()


# X represent the attributes and Y is the label(values are to be predicted)

X = dataset['MinTemp'].values.reshape(-1, 1)
Y = dataset['MaxTemp'].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

#main algorithm

regressor = LinearRegression()
regressor.fit(X_train, Y_train) 

print(regressor.intercept_)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

# Campare value generate from our algorithm and origin value
df = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

# df1 = df.head(25)
# df1.plot(kind='bar', figsize=(16,10))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()

plt.scatter(X_test, Y_test, color='gray')
plt.plot(X_test, y_pred, color='green', linewidth=2)
plt.show()

# Evaluate the performance of the algorithm:
print('Mean Absolute Error: ', metrics.mean_absolute_error(Y_test, y_pred))
print('Meann Squared Error: ', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))