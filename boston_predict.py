# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from pickle import load
import numpy as np

#Name of the saved model from another computer
filename = 'final_boston.sav'

#load the file
loaded_model = load(open(filename, 'rb'))


# define one new data instance for prediction
Xnew = [[0.01965,80,1.76,"0",0.385,6.23,31.5,9.0892,1,241,18.2,341.6,12.93]]
# convert data type of array to float64 
Xnew = np.array(Xnew, dtype=np.float64)

# make a prediction
ynew = loaded_model.predict(Xnew)
print("Input =%s, Predicted =%s" % (Xnew[0], ynew[0]))
