from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("titanicOriginalDataset.csv")
df=df.drop(["Name","Ticket","Cabin"],axis=1)
#print(df)5
df=pd.get_dummies(df,columns=["Pclass","Sex","Embarked"])
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)
normalized_df=(df-df.mean())/df.std()
import numpy as np
X=df.values
Y=df["Survived"].values
X=np.delete(X,1,axis=1)
X=np.delete(X,0,axis=1)
from sklearn.model_selection import train_test_split
avg_acc_KNN=0
for i in range(100):
    xTrain,xTest,yTrain,yTest =train_test_split(X,Y,test_size=.2,stratify=Y)
    clf_KNN = KNeighborsClassifier()
    clf_KNN.fit(xTrain, yTrain)
    acc_KNN = clf_KNN.score(xTest, yTest)
    avg_acc_KNN = avg_acc_KNN + acc_KNN
print(avg_acc_KNN)
