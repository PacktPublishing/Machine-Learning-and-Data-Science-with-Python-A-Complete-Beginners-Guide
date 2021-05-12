# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:51:43 2019

@author: abhilash
"""
from matplotlib import pyplot
#load the csv file using read_csv function of pandas library
from pandas import read_csv
filename = 'pima-indians-diabetes.csv'
#url = 'https://myfilecsv.com/test.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()