# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
#load the csv file using read_csv function of pandas library
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

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

kfold = KFold(n_splits = num_folds, random_state = seed)
model = LogisticRegression(solver='liblinear')

#save this model to disk for reuse
filename = 'pickle_model.sav'
dump(model,open(filename,'wb'))

# move this file to another computer or server

#load the file
loaded_model = load(open(filename, 'rb'))


results = cross_val_score(loaded_model, X, Y, cv=kfold)
print("Mean Estimated Accuracy Logistic Regression: %f " % (results.mean()))






