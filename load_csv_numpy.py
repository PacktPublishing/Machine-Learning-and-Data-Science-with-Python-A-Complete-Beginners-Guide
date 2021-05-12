# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 07:12:34 2019

@author: abhilash
"""

#load csv file using the numpy.loadtxt() function
from numpy import loadtxt
filename = 'pima-indians-diabetes.csv'
raw_data = open(filename, 'rb')
data = loadtxt(raw_data, delimiter=",")
print(data.shape)