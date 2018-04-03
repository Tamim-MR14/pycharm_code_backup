from sklearn import tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB





#reading data
train_df = pd.read_csv('Youtube01-Psy.csv')

cols_drop = ['COMMENT_ID', 'AUTHOR', 'DATE']
train_df = train_df.drop(cols_drop, axis=1)

#print(train_df)

train_target = train_df['CLASS'].values
train_data = train_df['CONTENT'].values

#print(type(train_data[0]))

X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.20)


##############################

tfidfVectorizer = TfidfVectorizer()
XTrainTfIdf = tfidfVectorizer.fit_transform(X_train)

clf = MultinomialNB().fit(XTrainTfIdf, y_train)

XTest = tfidfVectorizer.transform(X_test)
pred = clf.predict(XTest)
#print(pred)

acc = clf.score(XTest, y_test)
print(acc)

