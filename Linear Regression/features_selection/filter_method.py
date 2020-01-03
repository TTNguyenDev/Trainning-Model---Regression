import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

#Read dataset after reprocessing ussing quantile & zscore
dataset = pd.read_csv('student_quantileafterzscore.csv')
print(dataset.describe())

# Split data into 2 part:
# X: features, y: class label
X = dataset.iloc[:,0:17]
y = dataset.iloc[:,17].values

#Generate traininng set and test set with test size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

full_data = X_train.copy()
full_data['mpg']=y_train

print("\n\nOrigin data: ")
print(X_train)
print("\n\nPredicted data: ")
print(full_data)

#Features selection using filter methods
importances = full_data.drop("mpg", axis=1).apply(lambda x: x.corr(full_data.mpg))
indices = np.argsort(importances)
print(importances)

names = ['UniID', 'Sex', 'Age', 'Area', 'FaEdu', 'MoEdu', 'FaJob', 'MoJob', 'Reason', 'StudyTime', 'Failures', 'HigherEdu', 'Freetime', 'Friends', 'DAlc', 'WAlc', 'Absences']
plt.title('Miles Per Gallon')
plt.barh(range(len(indices)), importances[indices], color='r', align='center')
plt.yticks(range(len(indices)), [names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# first iteration of features selection
print("\nmpg > 0.18: ")
for i in range(0, len(indices)):
    if np.abs(importances[i]) > 0.18:
        print(names[i])

X_train = X_train[['UniID', 'Sex', 'Area', 'MoEdu', 'StudyTime']]
#second iteration of features selection
for i in range(0, len(X_train.columns)):
    for j in range(0, len(X_train.columns)):
        if i!=j:
            corr_1 = np.abs(X_train[X_train.columns[i]].corr(X_train[X_train.columns[j]]))
            if corr_1 < 0.3:
                print(X_train.columns[i], "is not correlared with", X_train.columns[j])
            elif corr_1 > 0.75:
                print(X_train.columns[i], "is highly correlared with", X_train.columns[j])


# third iteration of features selection
print(len(X_train), len(y_train))
mi = mutual_info_regression(X_train, y_train)

mi = pd.Series(mi)
mi.index = X_train.columns
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10,4))
plt.show()

#With this dataset, we choose 5 features that most important to increase model accuracy
X_train = X_train[['UniID', 'Sex', 'Area', 'MoEdu', 'StudyTime']]

# Using LinearRegression of SKLearn to train model
regressor = LinearRegression()
print("train")
print(X_train)
print(len(y_train))
regressor.fit(X_train, y_train)

Xs = X[['UniID', 'Sex', 'Area', 'MoEdu', 'StudyTime']]
coeff_df = pd.DataFrame(regressor.coef_, Xs.columns, columns=['Coefficient'])
print(coeff_df)
X_test = X_test[['UniID', 'Sex', 'Area', 'MoEdu', 'StudyTime']]
y_pred = regressor.predict(X_test)

print("XXXX")
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
print(df1)

df1.plot(kind='bar', figsize=(10,8))
print(dataset.describe())

# Evaluate the performance of the algorithm:
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('Meann Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Compare result without using feature selection
print("*"*80)
# #Read dataset
dataset = pd.read_csv('student_quantileafterzscore.csv')
del dataset['Failures']
del dataset['HigherEdu']

X = dataset.iloc[:,0:15]
y = dataset.iloc[:,15].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(X_test)
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('Meann Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))