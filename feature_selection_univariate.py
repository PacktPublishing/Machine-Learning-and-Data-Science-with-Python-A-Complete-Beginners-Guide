# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
#load the csv file using read_csv function of pandas library
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

filename = 'pima-indians-diabetes.csv'
#url = 'https://myfilecsv.com/test.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values

#splitting the array to input and output
X = array[:,0:8]
Y = array[:,8]

#feature selection
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)

#print the scores for the features
set_printoptions(precision=3)
print(fit.scores_)

#print the first five rows of the best 4 features (Columns) selected
features = fit.transform(X)
print(features[0:5,:])









