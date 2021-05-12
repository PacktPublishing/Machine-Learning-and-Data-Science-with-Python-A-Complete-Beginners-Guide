# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
#load the csv file using read_csv function of pandas library
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


filename = 'pima-indians-diabetes.csv'
#url = 'https://myfilecsv.com/test.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values

#splitting the array to input and output
X = array[:,0:8]
Y = array[:,8]

#RFE feature extraction using logistic regression
model = LogisticRegression(solver='liblinear')
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)

print("Number of features: %d" % fit.n_features_)
print("selected Features : %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)












