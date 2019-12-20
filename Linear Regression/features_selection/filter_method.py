import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

#Read dataset
dataset = pd.read_csv('/Users/ttnguyen/Desktop/AtSchool/Machine Learning/Linear Regression/Weather.csv')
print(dataset.head(2))
print(dataset.info())
