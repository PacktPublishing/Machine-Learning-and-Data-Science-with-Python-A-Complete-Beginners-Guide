# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 07:23:51 2019

@author: abhilash
"""


import numpy
import pandas

#series
myarray1 = numpy.array([1, 2, 3])
rownames = ['a', 'b', 'c']
myseries = pandas.Series(myarray1, index=rownames)
print(myseries)

print(myseries[0])
print(myseries['a'])

#Dataframe
myarray2 = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames2 = ['a', 'b']
colnames2 = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray2, index=rownames2, columns=colnames2)
print(mydataframe)

print("method 1: printing column one %s" % mydataframe['one'])
print("method 2: printing column one %s" % mydataframe.one)


