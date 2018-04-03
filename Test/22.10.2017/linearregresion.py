import sklearn as sk
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


dataset = load_boston()
XTrain, XTest, YTrain, YTest = train_test_split(dataset.data,dataset.target,test_size=0.2)

est = linear_model.LinearRegression().fit(XTrain,YTrain)
score=est.score(XTest,YTest)
YPred = est.predict(XTest)

tups = [(YTest[i],YPred[i]) for i in range(len(YTest))]
tups.sort(key=lambda tup: tup[0])

x = np.arange(len(YTest))
yo=[t[0] for t in tups]
yp=[t[1] for t in tups]
plt.plot(x,yo,'r')
plt.plot(x,yp,'g')
plt.show()
print(est.coef_)
print(est.intercept_)
