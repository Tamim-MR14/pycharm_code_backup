from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




iris=load_iris()
X=iris.data
Y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=iris.target)
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X_train,y_train)
print(sum(y_test==1))
print(clf.score(X_train,y_train)*100,clf.score(X_test,y_test)*100)