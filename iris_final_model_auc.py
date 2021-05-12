#from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from pandas import read_csv


#load the csv file using read_csv function of pandas library
filename = 'iris_encoded.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
#splitting the array to input and output
X = array[:,0:4]
y = array[:,4]

#iris = datasets.load_iris()
#X, y = iris.data, iris.target

y = label_binarize(y, classes=[0,1,2])
n_classes = 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=50)

clf = OneVsRestClassifier(LogisticRegression())
y_score = clf.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='ROC curve Class 0 vs Rest (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='ROC curve Class 1 vs Rest (area = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='ROC curve Class 2 vs Rest (area = %0.2f)' % roc_auc[2])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


