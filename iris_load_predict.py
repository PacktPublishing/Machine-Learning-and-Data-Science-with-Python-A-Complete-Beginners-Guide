# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from pickle import load

#Name of the saved model from another computer
filename = 'iris.sav'

#load the file
loaded_model = load(open(filename, 'rb'))


# define one new data instance for prediction
Xnew = [[6.2,3.1,5.2,2.4]]

# make a prediction
ynew = loaded_model.predict(Xnew)
ynew_prob = loaded_model.predict(Xnew)
print("Input =%s, Predicted =%s" % (Xnew[0], ynew[0]))
