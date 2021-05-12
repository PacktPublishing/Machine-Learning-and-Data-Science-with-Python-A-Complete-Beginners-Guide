# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 05:18:39 2019

@author: abhilash
"""

import numpy
mylist = [[1, 2, 3], [3, 4, 5]]
#convert the python list into numpy array
myarray = numpy.array(mylist)
print(myarray)
print(myarray.shape)

#get the contents of the numpy array
print("First Row: %s" % myarray[0])
print("Last Row: %s" % myarray[-1])

#get a specific value
print("specific row and col: %s" % myarray[0, 2])
#get the whole column
print("whole col: %s" % myarray[:, 2])

#arithmetic operations with numpy array 
myarray1 = numpy.array([2, 2, 2])
myarray2 = numpy.array([3, 3, 3])
print("Addition: %s" % (myarray1 + myarray2))
print("Multiplication: %s" % (myarray1 * myarray2))








