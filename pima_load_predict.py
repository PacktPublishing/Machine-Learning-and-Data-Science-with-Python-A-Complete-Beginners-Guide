# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from pickle import load

#Name of the saved model from another computer
filename = 'final_pima_indian.sav'

#load the file
loaded_model = load(open(filename, 'rb'))


# define one new data instance for prediction
Xnew = [[0,132,90,33,167,40.1,3.288,30]]

# make a prediction
ynew = loaded_model.predict(Xnew)
print("Input =%s, Predicted =%s" % (Xnew[0], ynew[0]))
