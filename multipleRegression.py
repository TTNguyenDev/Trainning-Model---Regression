import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('winequality.csv')

# print(dataset.shape, dataset.describe())

# print(dataset.isnull().any())

#optional: remove all the null values from cloumn
#dataset = dataset.fillna(method='ffill')

#divided 
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
Y = dataset['quality'].values

# check the average value of the “quality” column
# show chart of label column
# plt.figure(figsize=(15,10))
# plt.tight_layout()
# seabornInstance.distplot(dataset['quality'])
# plt.show()

X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Xs = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']]
coeff_df = pd.DataFrame(regressor.coef_, Xs.columns, columns=['Coefficient'])
print(coeff_df)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
print(df1)

df1.plot(kind='bar', figsize=(10,8))
print(dataset.describe())