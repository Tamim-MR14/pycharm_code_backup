from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#dropping unnecessary columns
cols_drop = ['Name', 'Ticket', 'Cabin']
train_df = train_df.drop(cols_drop, axis=1)

#handling missing embarked values
max_embark = train_df['Embarked'].dropna().max()
train_df['Embarked'] = train_df['Embarked'].fillna(max_embark)

#one-hot-encoding categorical values
dummies = []
cols_dummy = ['Pclass', 'Sex', 'Embarked']
for col in cols_dummy:
    dummies.append(pd.get_dummies(train_df[col]))

train_dummies = pd.concat(dummies, axis=1)
cols_drop_dummy = [3, 'male', 'S']
train_dummies = train_dummies.drop(cols_drop_dummy, axis=1)
#print(train_dummies)
train_df = pd.concat((train_df, train_dummies), axis=1)
train_df = train_df.drop(['Pclass', 'Sex', 'Embarked'], axis=1)

#We are going to deal missing age values in 4 different ways.
#1. Replace by a constant: -1
#2. Replace by average
#3. Replace by interpolation
#4. Reverse machine learning

#Method 1:
missing_age = -1
train_df['Age'] = train_df['Age'].fillna(missing_age)


print(train_df.info())

#seperating data and target
train_target = train_df['Survived'].values
train_data = train_df.drop('Survived', axis=1).values

#print(train_target)
#print(train_data)

#train_test_split and ML apply
avg_acc_tree = 0
avg_acc_perceptron = 0
avg_acc_KNN = 0
for i in range (0, 1000):
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


print('Accuracy for DecisionTree: {}'.format(avg_acc_tree/10))
print('Accuracy for Perceptron: {}'.format(avg_acc_perceptron/10))
print('Accuracy for KNN: {}'.format(avg_acc_KNN/10))




