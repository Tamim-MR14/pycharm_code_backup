import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import numpy as np
import random
a=np.random.uniform(0,1,500)
b=np.random.uniform(0,1,500)

d=a*b

data=np.array(list(zip(a,b)))
print(data)
total=0

for i in range(30):
    XTrain,XTest,YTrain,YTest=train_test_split(data,d,test_size=0.2)

    clf=MLPRegressor(hidden_layer_sizes=(50,50),activation='relu', solver='lbfgs')
    clf=clf.fit(XTrain,YTrain)
    out=clf.score(XTest,YTest)
    total+=out
print("Output on is:",total/30)