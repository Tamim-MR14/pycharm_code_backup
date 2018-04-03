from sklearn import tree
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import ensemble



#reading data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#dropping unnecessary columns
cols_drop = ['Ticket']
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

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].dropna().mean())

train_df['Name'] = train_df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
test_df['Name'] = test_df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
titles = train_df['Name'].unique()
titles2 = test_df['Name'].unique()
print(titles, titles2)

# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Officer",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Miss",
                    "Mlle":       "Miss",
                    "Ms":         "Miss",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
#titles = train_df['Name'].unique()
#print(titles)

# we map each title
train_df['Name'] = train_df['Name'].map(Title_Dictionary)
test_df['Name'] = test_df['Name'].map(Title_Dictionary)

#train_df['Name'] = LabelEncoder().fit_transform(train_df['Name'])
#test_df['Name'] = LabelEncoder().fit_transform(test_df['Name'])
#train_df['Name'] = StandardScaler().fit_transform(train_df['Name'].values.reshape(-1, 1))
#test_df['Name'] = StandardScaler().fit_transform(test_df['Name'].values.reshape(-1, 1))


#cabin:
train_df['Cabin'].fillna('U', inplace=True)
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: x[0])
test_df['Cabin'].fillna('U', inplace=True)
test_df['Cabin'] = test_df['Cabin'].apply(lambda x: x[0])

dummies_train = []
dummies_test = []
cols_dummy = ['Name']
for col in cols_dummy:
    dummies_train.append(pd.get_dummies(train_df[col]))
    dummies_test.append(pd.get_dummies(test_df[col]))

train_dummies = pd.concat(dummies_train, axis=1)
test_dummies = pd.concat(dummies_test, axis=1)
cols_drop_dummy = ['Master']
train_dummies = train_dummies.drop(cols_drop_dummy, axis=1)
test_dummies = test_dummies.drop(cols_drop_dummy, axis=1)
#print(train_dummies)
train_df = pd.concat((train_df, train_dummies), axis=1)
test_df = pd.concat((test_df, test_dummies), axis=1)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

#We are going to deal missing age values in 4 different ways.
#1. Replace by a constant: -1
#2. Replace by average
#3. Replace by interpolation
#4. Reverse machine learning

#Method 4:
ageless_df = train_df[pd.isnull(train_df['Age'])].drop(['Cabin'], axis=1)
print(ageless_df.info())
agewith_df = train_df[pd.notnull(train_df['Age'])].drop(['Cabin'], axis=1)
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

new_train_df['family size'] = new_train_df['SibSp'] + new_train_df['Parch'] + 1

dummiesc_train = []
dummiesc_test = []
cols_dummy = ['Cabin']
for col in cols_dummy:
    dummiesc_train.append(pd.get_dummies(train_df[col]))
    dummiesc_test.append(pd.get_dummies(test_df[col]))

train_dummiesc = pd.concat(dummiesc_train, axis=1)
test_dummiesc = pd.concat(dummiesc_test, axis=1)
cols_drop_dummyc = ['F']
train_dummiesc = train_dummiesc.drop(cols_drop_dummyc, axis=1)
test_dummiesc = test_dummiesc.drop(cols_drop_dummyc, axis=1)
#print(train_dummies)
new_train_df = pd.concat((new_train_df, train_dummiesc), axis=1)
#test_df = pd.concat((test_df, test_dummies), axis=1)
#train_df = train_df.drop(['Name'], axis=1)
#test_df = test_df.drop(['Name'], axis=1)


print(new_train_df)

#seperating data and target
train_target = new_train_df['Survived'].values
train_data = new_train_df.drop(['Survived', 'PassengerId'], axis=1).values
#test_data = new_test_df.drop(['Survived', 'PassengerId'], axis=1).values

#print(train_target)
#print(train_data)

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
