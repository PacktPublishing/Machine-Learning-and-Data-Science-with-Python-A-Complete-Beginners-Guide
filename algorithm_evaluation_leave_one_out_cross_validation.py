# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
#load the csv file using read_csv function of pandas library
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.csv'
#url = 'https://myfilecsv.com/test.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values

#splitting the array to input and output
X = array[:,0:8]
Y = array[:,8]

num_folds = 10
seed = 7

loocv = LeaveOneOut()
model = LogisticRegression(solver='liblinear')

results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: %.3f%% (%.3f%%) " % (results.mean()*100.0, results.std()*100.0))






