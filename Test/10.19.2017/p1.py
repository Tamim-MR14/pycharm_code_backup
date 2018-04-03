from sklearn import tree
from sklearn import neighbors
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
Y = iris.target
eff1=[]
for j in range(50):
    eff=0
    for i in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        clf = neighbors.KNeighborsClassifier(n_neighbors=(2*j+1))
        clf = clf.fit(X_train, y_train)
        acc=clf.score(X_test, y_test) * 100
        eff=eff+acc

    eff=eff/30
    eff1.append(eff)
    print("For k=",(2*j+1),"The average eff is",eff,"%")

k=[2*i+1 for i in range(50)]
plt.plot(k,eff1)
plt.show()