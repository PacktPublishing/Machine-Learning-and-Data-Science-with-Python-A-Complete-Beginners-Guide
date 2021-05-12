# Bagging Algorithms
# Importing Bagged Decision Trees, Random Forest, Extra Trees
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

seed = 7

num_trees = 100
max_features = 3

kfold = KFold(n_splits=10, random_state=seed)

# Bagged Decision Trees Model
cart = DecisionTreeClassifier()
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy Bagged Decision Trees: %f " % (results.mean()))

# Random Forest Model
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy Random Forest: %f " % (results.mean()))

# Extra Trees Model
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy Extra Trees: %f " % (results.mean()))