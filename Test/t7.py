fo=open("test.csv",'r+')

x=fo.readline()
pcount=0
while True:
    x = fo.readline()
    if x=='':
        break

    x1=x.split(',')
    print(x1)
    if (2*float(x1[0])-float(x1[1])-1)<0:
        if x1[2].strip()=='A':
            pcount+=1
    else:
        if x1[2].strip()=='B':
            pcount+=1

print(pcount)

