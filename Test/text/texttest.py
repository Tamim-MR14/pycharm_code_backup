from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

df=pd.read_csv('youtube01-Psy.csv')
cols_drop=['COMMENT_ID', 'AUTHOR','DATE']
df=df.drop(cols_drop,axis=1)
target=df['CLASS'].values
data=df['CONTENT'].values
X_train,X_test,Y_train,Y_test=train_test_split(data,target, test_size=0.2)
tfidfVectorizer=TfidfVectorizer()
X_train_tfidf=tfidfVectorizer.fit_transform(X_train)
clf=MultinomialNB().fit(X_train_tfidf,Y_train)

X_test_tfidf=tfidfVectorizer.transform(X_test)
print(clf.score(X_test_tfidf,Y_test))
ann=MLPClassifier(hidden_layer_sizes=(50,50),activation='relu', solver='lbfgs',max_iter=200)
ann.fit(X_train_tfidf,Y_train)
print(ann.score(X_test_tfidf,Y_test))

knn=KNeighborsClassifier(n_neighbors=5, leaf_size=30, p=2, n_jobs=1)
knn.fit(X_train_tfidf,Y_train)
print(knn.score(X_test_tfidf,Y_test))

DT=DecisionTreeClassifier( )
DT.fit(X_train_tfidf,Y_train)
print(DT.score(X_test_tfidf,Y_test))