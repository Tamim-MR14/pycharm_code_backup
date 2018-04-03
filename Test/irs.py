foo=open('iris.csv')
temp=[]
foo.readline()
for i in range(0,150):
    x=foo.readline()
    x=x.split(",")
    tempList=[]
    for i in range(0,4):
        tempList.append(float(x[i]))
    tempList.append(str.strip(x[4]))
    temp.append(tempList)

import random
def knn(tr,t):
    d = []
    for i in range(len(tr)):
        y = ((t[0] - tr[i][0]) ** 2 + (t[1] - tr[i][1]) ** 2 + (t[2] - tr[i][2]) ** 2 + (t[3] - tr[i][3]) ** 2) ** 0.5
        d.append(y)
    d2 = d.copy()
    d2 = sorted(d2)
    setosa=0
    versicolor=0
    verginica=0
    for i in range(len(tr)):
        if d[i]==d2[0] or d[i]==d2[1] or d[i]==d2[2] or d[i]==d2[3] or d[i]==d2[4]:
            if train[i][4]=='Iris-setosa':
                setosa+=1
            elif train[i][4]=='Iris-versicolor':
                versicolor+=1
            elif train[i][4]=='Iris-virginica':
                verginica+=1
    if max(setosa,versicolor,verginica)==setosa:
            label='Iris-setosa'
            return label
    elif max(setosa,versicolor,verginica)==versicolor :
            label='Iris-versicolor'
            return label
    elif max(setosa,versicolor,verginica)==verginica :
            label='Iris-virginica'
            return label
eff=0

for i in range(0,30):
   train=temp.copy()
   random.shuffle(train)
   test=train[120:150]
   temp1=train[0:120]
   train=temp1
   correct=0
   for j in range(len(test)):
       lb=knn(train, test[j])
       if test[j][4]==lb:
           correct+=1

   acc=(correct/len(test))*100
   print(acc)
   eff=eff+acc

eff=eff/30
print('The final avarage is',eff,'%')