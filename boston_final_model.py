# -*- coding: utf-8 -*-
"""
@author: abhilash
"""

from pandas import read_csv
from sklearn.linear_model import LinearRegression
from pickle import dump

#load the csv file using read_csv function of pandas library
filename = 'BostonHousing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
dataframe = read_csv(filename, names=names)

array = dataframe.values
#splitting the array to input and output
X = array[:,0:13]
Y = array[:,13]

######## DO REQUIRED SUMMARIZATIONS, EVALUATIONS and OPTIMIZATIONS HERE #######

model = LinearRegression()
model.fit(X,Y)


#save this model to disk for reuse
filename = 'final_boston.sav'
dump(model,open(filename,'wb'))