foo=open("iris.csv")
temp=[]
foo.readline()
for i in range(0,150):
    x=foo.readline()
    x=x.split(",")
    tempList=[]
    for i in range(0,4):
        tempList.append(float(x[i]))
    tempList.append(x[4])
    temp.append(tempList)

print(temp[0])
ls=[5.8,2.7,5.1,1.9]
d=[]
for i in range(0,150):
   y=((ls[0]-temp[i][0])**2+(ls[1]-temp[i][1])**2+(ls[2]-temp[i][2])**2+(ls[3]-temp[i][3])**2)**0.5
   d.append(y)
print(d)
d2=d.copy()
d2=sorted(d2)
print(d2)
hudai=0
for i in range(0,150):
    if d[i]==d2[0] or d[i]==d2[1] or d[i]==d2[2] or d[i]==d2[3] or d[i]==d2[4]:
        print(temp[i])
        print(i)

