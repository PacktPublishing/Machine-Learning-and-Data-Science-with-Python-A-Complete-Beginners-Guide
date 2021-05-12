# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
#load the csv file using read_csv function of pandas library
from pandas import read_csv
from sklearn.decomposition import PCA

filename = 'pima-indians-diabetes.csv'
#url = 'https://myfilecsv.com/test.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values

#splitting the array to input and output
X = array[:,0:8]
Y = array[:,8]

#feature selection using PCA
pca = PCA(n_components=4)
fit = pca.fit(X)

#summarize the components
print("Explained Varience Ratio Summary: %s" % fit.explained_variance_ratio_)
print(fit.components_)








