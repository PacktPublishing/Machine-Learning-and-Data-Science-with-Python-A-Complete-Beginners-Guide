# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
#load the csv file using read_csv function of pandas library
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Binarizer

filename = 'pima-indians-diabetes.csv'
#url = 'https://myfilecsv.com/test.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values

#splitting the array to input and output
X = array[:,0:8]
Y = array[:,8]

binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)

set_printoptions(precision=3)
print(binaryX[0:5,:])