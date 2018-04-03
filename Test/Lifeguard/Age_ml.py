from sklearn import tree
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn.model_selection import cross_val_score




#reading data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#dropping unnecessary columns
cols_drop = ['Name', 'Ticket', 'Cabin']
train_df = train_df.drop(cols_drop, axis=1)
test_df = test_df.drop(cols_drop, axis=1)

#handling missing embarked values
max_embark = train_df['Embarked'].dropna().max()
train_df['Embarked'] = train_df['Embarked'].fillna(max_embark)

#creating new feature
train_df['new'] = (train_df['Pclass']==1) & (train_df['Sex']=='female')
test_df['new'] = (test_df['Pclass']==1) & (test_df['Sex']=='female')

#one-hot-encoding categorical values
train_df['Sex'] = LabelEncoder().fit_transform(train_df['Sex'])
train_df['Pclass'] = LabelEncoder().fit_transform(train_df['Pclass'])
train_df['Embarked'] = LabelEncoder().fit_transform(train_df['Embarked'])
train_df['new'] = LabelEncoder().fit_transform(train_df['new'])

test_df['Sex'] = LabelEncoder().fit_transform(test_df['Sex'])
test_df['Pclass'] = LabelEncoder().fit_transform(test_df['Pclass'])
test_df['Embarked'] = LabelEncoder().fit_transform(test_df['Embarked'])
test_df['new'] = LabelEncoder().fit_transform(test_df['new'])

#We are going to deal missing age values in 4 different ways.
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
agewith_data = agewith_df.drop(['Survived', 'PassengerId', 'Age'], axis=1).values
ageless_data = ageless_df.drop(['Survived', 'PassengerId', 'Age'], axis=1).values

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
max_depth_range = range(2,8)
tree_accuracy = []

for i in max_depth_range:
    clf_tree = tree.DecisionTreeClassifier(max_depth=i)
    accuracy = cross_val_score(clf_tree, train_data, train_target, cv = 10, scoring='accuracy')
    tree_accuracy.append(accuracy.mean())

print(tree_accuracy)

plt.plot(max_depth_range, tree_accuracy)
plt.show()