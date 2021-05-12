# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 07:22:30 2019

@author: abhilash
"""

#load the csv file using read_csv function of pandas library
from pandas import read_csv
from pandas import set_option
filename = 'pima-indians-diabetes.csv'
#url = 'https://myfilecsv.com/test.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

set_option('display.width', 200)
set_option('display.max_columns', 10)
set_option('precision', 3)

description = data.describe()
print(description)

