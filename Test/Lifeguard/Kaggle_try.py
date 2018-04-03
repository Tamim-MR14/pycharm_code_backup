from sklearn import tree
from sklearn.linear_model import Perceptron
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
print(test_df.info())

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

#Method 2:
mean_age_train = train_df['Age'].mean()
train_df['Age'] = train_df['Age'].fillna(mean_age_train)
mean_age_test = test_df['Age'].mean()
test_df['Age'] = test_df['Age'].fillna(mean_age_test)

test_df['Fare'] = test_df['Fare'].fillna(-1)

print(train_df.info())
print(test_df.info())

#seperating data and target
train_target = train_df['Survived'].values
train_data = train_df.drop(['Survived', 'PassengerId'], axis=1).values
test_data = test_df.drop(['PassengerId'], axis=1).values

#print(train_target)
print(train_data)

#train_test_split and ML apply
#clf_tree = tree.DecisionTreeClassifier(max_depth=4)
#clf_tree.fit(train_data, train_target)

clf_forest = ensemble.RandomForestClassifier(n_estimators=100)
clf_forest.fit(train_data, train_target)

#pred_tree = clf_tree.predict(test_data)
pred_forest = clf_forest.predict(test_data)

idvalues = np.arange(892, 1310)
output = np.column_stack((idvalues, pred_forest))
df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
df_results.to_csv('titanic_results_forest.csv',index=False)





