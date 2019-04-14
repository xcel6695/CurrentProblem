# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 22:21:29 2019

@author: Veer
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values ""
Y = dataset.iloc[:, 3].values
print (X, Y)

# Handle Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print (X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X = LabelEncoder()
X[:, 0] = LabelEncoder_X.fit_transform(X[:, 0])
print(X)
OneHotEncoder_X = OneHotEncoder(categorical_features = [0])
X = OneHotEncoder_X.fit_transform(X).toarray()
print(X)
LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(Y)
print(y)

from sklearn.cross_validation import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.2, random_state = 0)  

from sklearn.preprocessing import StandardScaler
SC_X = StandardScaler()
X_Train = SC_X.fit_transform(X_Train)
X_Test = SC_X.fit_transform(X_Test)



