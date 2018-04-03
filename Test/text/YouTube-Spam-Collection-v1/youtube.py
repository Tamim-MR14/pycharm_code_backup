from setuptools.command.test import test
from sklearn import tree
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

trainDocs = pd.read_csv("Youtube01-Psy.csv",header=0)
#drop some of the columns
cols = ['COMMENT_ID','AUTHOR','DATE']
trainDocs = trainDocs.drop(cols,axis=1)
data=trainDocs.CONTENT
target=trainDocs.CLASS

vectorizer = CountVectorizer()
data = vectorizer.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data,target,test_size=0.3,random_state=0)

clf = MultinomialNB().fit(X_train, y_train)

# XTest = tfidfVectorizer.transform(testDocs)
score = clf.score(X_test,y_test)
print(score)
# testLabels = [ClassNames[i] for i in pred]
# print(testLabels)
#
# print(X_train)

