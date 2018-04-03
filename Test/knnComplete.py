import random
#starting
foo=open("iris.csv")
originalSequence=[]
foo.readline()

for i in range(0,150):
    x=foo.readline()
    x=x.split(",")
    tempList=[]
    for i in range(0,4):
        tempList.append(float(x[i]))
    tempList.append(x[4])
    originalSequence.append(tempList)

print("original sequence\n",originalSequence)

rightArray=[]

for m in range(0,100):
    trainVector=originalSequence.copy()
    random.shuffle(trainVector)
    #print("shuffled sequence\n",trainVector)
    temporary=trainVector[0:120]
    testVector=trainVector[120:150]
    trainVector=temporary
    #train test e vag hoiya gese randomly
    right=0
    wrong=0
    #print(testVector)

    lenTV=len(trainVector)
    list=[]
    for j in range(0,len(testVector)):
        irisVirginica = 0
        irisSetosta = 0
        irisVersicolor = 0
        ls=testVector[j]
        d=[]
        for i in range(0,lenTV):
           y=((ls[0]-trainVector[i][0])**2+(ls[1]-trainVector[i][1])**2+(ls[2]-trainVector[i][2])**2+(ls[3]-trainVector[i][3])**2)**0.5
           d.append(y)
        #print(d)
        d2=d.copy()
        d2=sorted(d2)
        #print(d2)

        for i in range(0,lenTV): #TrainVector = 120
            if d[i]==d2[0] or d[i]==d2[1] or d[i]==d2[2] or d[i]==d2[3] or d[i]==d2[4]:
                #print(trainVector[i])
                #print(i)
                if trainVector[i][4]=="Iris-versicolor\n":
                    irisVersicolor+=1
                elif trainVector[i][4] == "Iris-setosa\n":
                    irisSetosta += 1
                elif trainVector[i][4] == "Iris-virginica\n":
                    irisVirginica += 1

        if max(irisVirginica,irisVersicolor,irisSetosta)==irisSetosta and testVector[j][4]=="Iris-setosa\n":
            right+=1
        elif max(irisVirginica,irisVersicolor,irisSetosta)==irisVersicolor and testVector[j][4]=="Iris-versicolor\n":
            right+=1
        elif max(irisVirginica,irisVersicolor,irisSetosta)==irisVirginica and testVector[j][4]=="Iris-virginica\n":
            right+=1

    """
    #bujhar jonno
        if irisSetosta>irisVirginica and irisSetosta >irisVersicolor:
            print("original is ",testVector[j][4],"assumption is irisSentosa\n\n")
        elif irisVirginica>irisSetosta and irisVirginica >irisVersicolor:
            print("original is ",testVector[j][4],"assumption is irisVerginica\n\n")
        elif irisVersicolor > irisSetosta and irisVersicolor > irisVirginica:
            print("original is ", testVector[j][4], "assumption is irisVersicolor\n\n")
    """

        #print(newLs)
        #print(testVector[j][4])
        #list.append(newLs)
    #print(list)
    print(right)
    efficiency=right/len(testVector)
    print(efficiency*100,"%")
    rightArray.append(right)
print(rightArray)
average = sum(rightArray)/len(rightArray)
print("final efficiency is ",(average/len(testVector))*100,"%" )
