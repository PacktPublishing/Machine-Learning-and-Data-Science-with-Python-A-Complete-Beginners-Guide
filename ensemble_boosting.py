# Boosting Algorithms
# Importing AdaBoostClassifier and GradientBoostingClassifier
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age','class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 100
kfold = KFold(n_splits=10, random_state=seed)

# ADA Boost Model
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed) 
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy ADA Boost: %f " % (results.mean()))

# Gradient Boosting Model
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy Gradient Boost: %f " % (results.mean()))
