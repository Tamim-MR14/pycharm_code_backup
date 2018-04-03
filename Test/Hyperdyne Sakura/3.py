import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neural_network import MLPRegressor

training = []
test = []
df = pd.read_csv("modified tokyo.csv")
k = 0
m = 0
for i in range(1961, 2018):

    bdi = df.loc[(df['year'] == i) & df['bloom'].isin([1])].index.tolist()
    df2 = df.loc[(df['year'] == i) & (df['serial'] <= bdi[0])]
    df2.index = range(len(df2))

    if i in [1966, 1971, 1985, 1994, 2008]:
        test.append([])
        for j in [4, 5, 12, 13]:
            test[k].append(df2.iloc[:, j].mean())
        for j in [9, 10, 11, 14]:
            test[k].append(df2.iloc[:, j].sum())
        test[k].append(len(df2))
        k += 1
        continue

    training.append([])
    for j in [4, 5, 12, 13]:
        training[m].append(df2.iloc[:, j].mean())
    for j in [9, 10, 11, 14]:
        training[m].append(df2.iloc[:, j].sum())
    training[m].append(len(df2))
    m += 1

# print(test)
print(len(test), len(test[0]))
print(len(df2))
# print(training)
# print(len(training),len(training[0]))
print(df2)
#X_train=np.array(training)[:,:8]
#Y_train=np.array(training)[:,-1:]
#print(X_train)
#print(len(X_train),len(X_train[0]))
#print(Y_train)
#print(len(Y_train),len(Y_train[0]))

#X_test=np.array(test)[:,:8]
#Y_test=np.array(test)[:,-1:]
#print(X_test)
#print(len(X_test),len(X_test[0]))
#print(Y_train)
#print(len(Y_test),len(Y_test[0]))

clf=MLPRegressor(hidden_layer_sizes=(10,60),activation='relu', solver='lbfgs')
clf=clf.fit(X_train,Y_train)
out=clf.score(X_test,Y_test)
print(out)
