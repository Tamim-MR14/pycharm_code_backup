import pandas as pd
df=pd.read_csv("titanicOriginalDatasetCopy.csv")
df=df.drop(["Name","Ticket","Cabin"],axis=1)
#print(df)
df=pd.get_dummies(df,columns=["Pclass","Sex","Embarked"])
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)
normalized_df=(df-df.mean())/df.std()
#print(df4)
################################
#####################################

import numpy as np
X=df.values
Y=df["Survived"].values
#print(X)
newX=np.delete(X,1,axis=1) #column borabor (axis=1) row 1 kete felbe
#X just ekta numpy array ekhon
X=newX
#print(X[1])
from sklearn.model_selection import train_test_split
sum=0
for i in range(100):
    xTrain,xTest,yTrain,yTest =train_test_split(X,Y,test_size=.2,stratify=Y)
    from sklearn import tree
    clf=tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(xTrain,yTrain)
    #print("efficiency for trainDataset ",clf.score(xTrain,yTrain))
    #print("efficiency for testDataset ",clf.score(xTest,yTest))
    sum=sum+clf.score(xTest,yTest)
average=sum/100
print(average)


#last bar er ta mone rakhbe