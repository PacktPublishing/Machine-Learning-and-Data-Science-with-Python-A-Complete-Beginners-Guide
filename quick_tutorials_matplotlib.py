# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 07:01:01 2019

@author: abhilash
"""

#plot a basic line plot
import matplotlib.pyplot as plt
import numpy

myarray = numpy.array([1, 2, 3])
plt.plot(myarray)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()

myarray1 = numpy.array([1, 2, 3])
myarray2 = numpy.array([1, 2, 3])
plt.scatter(myarray1, myarray2)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()