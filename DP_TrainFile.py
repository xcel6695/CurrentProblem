# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 22:21:29 2019

@author: Veer
"""
"importing required libraries"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train_submissions.csv')
"divding dependent and independent variables"
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 2].values
print (X)
print (Y)

"Assigning labels to independent variables"
from sklearn.preprocessing import LabelEncoder
LabelEncoder_X = LabelEncoder()
X[:, 0] = LabelEncoder_X.fit_transform(X[:, 0])
"Converted to dataframe as X cannot be seen in variable explorer"
"No need to do this for Y as it contains single column"
df_X = pd.DataFrame(X) 
print(X)

"""Dependent variable is already labelled
LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(Y)
print(y)"""

"To scale data on specific range"
from sklearn.preprocessing import StandardScaler
SC_X = StandardScaler()
df_X = SC_X.fit_transform(df_X)
Y = SC_X.fit_transform(Y)

"""Pending
1. Check warning on data scale section - DeprecationWarning: 
Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. 
Reshape your data either using X.reshape(-1, 1) if your data has a single feature or 
X.reshape(1, -1) if it contains a single sample
2.Import data with column names 
3. Writing data frame to csv file - 
Warning - 'numpy.ndarray' object has no attribute 'to_csv'
"""



