# -*- coding: utf-8 -*-
"""
@author: abhilash
"""

from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from pickle import dump

#load the csv file using read_csv function of pandas library
filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values
#splitting the array to input and output
X = array[:,0:4]
Y = array[:,4]

######## DO REQUIRED SUMMARIZATIONS, EVALUATIONS and OPTIMIZATIONS HERE #######
# class distribution
#print(dataset.groupby('class').size())
model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
model.fit(X,Y)

#save this model to disk for reuse
filename = 'iris.sav'
dump(model,open(filename,'wb'))
