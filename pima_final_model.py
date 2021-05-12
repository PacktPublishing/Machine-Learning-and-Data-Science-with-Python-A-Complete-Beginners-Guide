# -*- coding: utf-8 -*-
"""
@author: abhilash
"""

from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from pickle import dump

#load the csv file using read_csv function of pandas library
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values
#splitting the array to input and output
X = array[:,0:8]
Y = array[:,8]

######## DO REQUIRED SUMMARIZATIONS, EVALUATIONS and OPTIMIZATIONS HERE #######


model = LogisticRegression(solver='liblinear')
model.fit(X,Y)

#save this model to disk for reuse
filename = 'final_pima_indian.sav'
dump(model,open(filename,'wb'))

