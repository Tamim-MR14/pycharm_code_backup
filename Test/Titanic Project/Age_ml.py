from sklearn import tree
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble



#reading data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#dropping unnecessary columns
cols_drop = ['Ticket', 'Cabin']
train_df = train_df.drop(cols_drop, axis=1)
test_df = test_df.drop(cols_drop, axis=1)
train_df['Name'] = train_df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
test_df['Name'] = test_df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
#handling missing embarked values
max_embark = train_df['Embarked'].dropna().max()
train_df['Embarked'] = train_df['Embarked'].fillna(max_embark)

#creating new feature
train_df['new'] = (train_df['Pclass']==1) & (train_df['Sex']=='female')
test_df['new'] = (test_df['Pclass']==1) & (test_df['Sex']=='female')

#one-hot-encoding categorical values
train_df=pd.get_dummies(train_df,columns=["Pclass","Sex","Embarked","new","Name"])
test_df=pd.get_dummies(test_df,columns=["Pclass","Sex","Embarked","new","Name"])
#We are going to deal missing age values in 4 different ways
#1. Replace by a constant: -1
#2. Replace by average
#3. Replace by interpolation
#4. Reverse machine learning

#Method 4:
ageless_df = train_df[pd.isnull(train_df['Age'])]
print(ageless_df.info())
agewith_df = train_df[pd.notnull(train_df['Age'])]
print(agewith_df.info())

agewith_target = agewith_df['Age'].values
agewith_data = agewith_df.drop(['Survived', 'PassengerId','Age'], axis=1).values
ageless_data = ageless_df.drop(['Survived', 'PassengerId','Age'], axis=1).values

clf_age = LinearRegression()
clf_age.fit(agewith_data, agewith_target)
pred_age = clf_age.predict(ageless_data)

ageless_df['Age'] = pred_age

new_train_df = pd.concat((ageless_df,agewith_df))
new_train_df = new_train_df.sort_values('PassengerId')
print(new_train_df)

#seperating data and target
train_target = new_train_df['Survived'].values
train_data = new_train_df.drop(['Survived', 'PassengerId'], axis=1).values

#print(train_target)
print(train_data)

#train_test_split and ML apply
avg_acc_tree = 0
avg_acc_perceptron = 0
avg_acc_KNN = 0
for i in range(0, 1000):
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.20)

    clf_tree = tree.DecisionTreeClassifier(max_depth=4)
    clf_tree.fit(X_train, y_train)

    clf_perceptron = Perceptron()
    clf_perceptron.fit(X_train, y_train)

    clf_KNN = KNeighborsClassifier()
    clf_KNN.fit(X_train, y_train)

    acc_tree = clf_tree.score(X_test, y_test)
    avg_acc_tree = avg_acc_tree + acc_tree

    acc_perceptron = clf_perceptron.score(X_test, y_test)
    avg_acc_perceptron = avg_acc_perceptron + acc_perceptron

    acc_KNN = clf_KNN.score(X_test, y_test)
    avg_acc_KNN = avg_acc_KNN + acc_KNN

print('Accuracy for DecisionTree: {}'.format(avg_acc_tree / 10))
print('Accuracy for Perceptron: {}'.format(avg_acc_perceptron / 10))
print('Accuracy for KNN: {}'.format(avg_acc_KNN / 10))








