from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
Y = iris.target
eff=0
for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf = svm.SVC()
    clf = clf.fit(X_train, y_train)
    acc=clf.score(X_test, y_test) * 100
    eff=eff+acc

eff=eff/30
print(eff)

