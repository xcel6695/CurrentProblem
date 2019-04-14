# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 22:21:29 2019

@author: Veer
"""
"importing required libraries"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('problem_data.csv')
"As all are independent variables, so we can skip the divide step"
X = dataset.iloc[:, :4].values

"Handle Missing data for numeric column (points column) - Replacing with mean" 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])
print (X)
"Converted to dataframe as X cannot be seen in variable explorer"
df_X = pd.DataFrame(X) 

"""Check how to handle Missing data for categorical columns ('Level Type', 'Tags' column) 
- Replacing with mode"""


"Assigning labels to required variables"
from sklearn.preprocessing import LabelEncoder
LabelEncoder_X = LabelEncoder()
X[:, 0] = LabelEncoder_X.fit_transform(X[:, 0])

print(X)

"Write df_X - not working as of now"
export_csv = df_X.to_csv (r'E:\Study\Current Problem\train\DP_train_X.csv', index = None, header=True)

"""Pending
1. Check categorical data to be labelled 
2. How should we replace NaN 
3. Writing data frame to csv file - 
Warning - 'numpy.ndarray' object has no attribute 'to_csv'
"""



