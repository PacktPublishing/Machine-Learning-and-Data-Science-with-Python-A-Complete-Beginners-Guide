# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 07:22:30 2019

@author: abhilash
"""

#load the csv file using read_csv function of pandas library
from pandas import read_csv
filename = 'pima-indians-diabetes.csv'
#url = 'https://myfilecsv.com/test.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# print a quick preview of first 20 rows
peek = data.head(20)
print(peek)
