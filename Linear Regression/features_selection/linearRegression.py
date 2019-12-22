import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


#Read dataset
dataset = pd.read_csv('student_quantileafterzscore.csv')

#predict max temp using all feature from column 1 to 25
dataset = dataset[['UniID', 'Sex', 'Area', 'MoEdu', 'StudyTime', 'Result']]
print(dataset)

X = dataset.iloc[:,0:4]
y = dataset.iloc[:,5].values

n_slpit = 10

MAE = []
MSE = []
RMSE = []

#Cross validation
kf = KFold(n_splits=n_slpit)
for train, test in kf.split(dataset):
    X_train, X_test, y_train, y_test = X.iloc[train,:], X.iloc[test,:], y[train], y[test]
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    MAE.append(metrics.mean_absolute_error(y_test, y_pred))
    MSE.append(metrics.mean_squared_error(y_test, y_pred))
    RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Mean Absolute Error: ', np.mean(MAE))
print('Meann Squared Error: ', np.mean(MSE))
print('Root Mean Squared Error: ', np.mean(RMSE))


#Build model:
X = dataset.iloc[:,0:4]
y = dataset.iloc[:,5].values

regressor = LinearRegression()
regressor.fit(X, y)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

