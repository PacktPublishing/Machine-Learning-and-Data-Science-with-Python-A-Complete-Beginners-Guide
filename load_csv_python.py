# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 06:33:04 2019

@author: abhilash
"""

#load csv file using standard python library
#include the module csv and use the function called reader()

import csv
import numpy

filename = 'pima-indians-diabetes.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')
print(data.shape)
