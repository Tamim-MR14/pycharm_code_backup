#problem-a
'''a=int(input("Enter the height of the triangle:"))
for i in range(1,a+1):
    for j in range(5):
        print('*',end='')
    print('\n')
for i in range(1,a+1):
    print('*'*a)'''

#Problem-b

'''a=int(input("Enter the height of the triangle:"))

for i in range(1,a+1):
    for j in range(i):
        print('*',end='')
    print("\n")
#alternate
for i in range(1,a+1):
    print('*'*i)'''

#Problem-c

'''a=int(input("Enter the heigt of the triangle:"))

for i in range(1,a+1):
    for j in range(1,a-i+2):
          print(" ",end='')

    for k in range(1,i+1):
        print("*",end='')
    print(' ')
#Alternate
for i in range(1,a+1):
    print(" "*(a-i+1),"*"*i)'''

#Problem-d

'''a=int(input("Enter the height of the triangle:"))

for i in range(1,a+1):
    for j in range(1,a-i+2):
        print('*',end='')
    for k in range(1,i):
        print(' ',end='')
    print('\n')

#alternate

for i in range(1,a+1):
    print("*"*(a-i+1),' '*(i-1))'''

#Problem-e

'''a=int(input("Enter the height of the triangle:"))

for i in range(1,a+1):
    for j in range(1,i):
        print(' ',end='')
    for k in range(1,a-i+2):
        print('*',end='')
    print('\n')

#alternate

for i in range(0,a):
    print(" "*i,end='')
    print("*"*(a-i))'''

#Problem-f
'''a=int(input("Enter the height of the triangle"))
for i in range(1,a+1):
    for j in range(1,a-i+2):
        print(" ",end='')
    for k in range(1,i+1):
        print('*',end='')
    for l in range(1,i):
        print('*',end='')
    print('\n')

for i in range(1,a+1):
    print(' '*(a-i),end='')
    print('*'*(2*i-1),end='')
    print('\n')'''

#Problem-2

'''a=input("Enter a binary number:")
d=0
b=len(a)
for i in range(0,b):
    d=d+(int(a[i])*2**(b-i-1))

print(d)'''

#problem-3

'''a=input("Enter a string:")
b=len(a)
for i in range(b):
    print(a[b-i-1],end='')'''

#Problem-4

'''a=input("Enter a string for Pelindrome test:")
b=len(a)
for i in range(0,b//2):
    if a[i]!=a[b-i-1]:
        print("Not Pelindrome")
        break
else:
    print("Pelindrome")'''

#Problem-5

'''a=int(input("Enter a integer for prime number test: "))

for i in range(2,a):
    if a%i==0:
        print("Not Prime")
        break
else:
    print("Prime")'''

#problem-6

'''a=int(input("Enter the value of n: "))

for i in range(2,a+1):
    for j in range(2,i):
        if i%j==0:
            break
    else:
        print(i,end=' ')'''

#problem-f1
'''fo=open("Hello.txt",'w')
fo.write("Hello Python")
fo.close()
fo=open("Hello.txt",'r+')
print(fo.readline())'''

#Problem-f2

'''fo=open("dummy.txt",'r+')

while True:
    x=fo.readline()
    if x=='':
        break
    x1=x.split('.')
    for i in range(len(x1)):
        print(x1[i])
        print('\n')
fo.close()'''

#Problem-f3

'''fo=open("marks.csv",'r+')
x=fo.readline()
ct1=0
ct2=0
count=0
while True:
    x=fo.readline()
    if x=='':
        break
    x1=x.split(',')
    ct1=ct1+int(x1[1])
    ct2 = ct2 + int(x1[2])
    count+=1
print(count)
print("Avereage CT1 Marks are: ",(ct1)/count)'''
#Problem-f4

'''fo=open("marks.csv",'r+')
x=fo.readline()
x1=[]

while True:
    x=fo.readline()
    if x=='':
        break
    x2=x.split(',\t')
    for i in range(len(x2)):
        x1.append(x2[i])
print(int(x1[2]))
print(x1)'''



class Point:

    def __init__(self,*t):
        if len(t)>0:
            self.x=float(t[0])
            self.y=float(t[1])
        else:
            self.x=0.0
            self.y=0.0

    def print(self):
        print('(',self.x,self.y,')')

p1=Point(2,3)
p1.print()
p2=Point()
p2.print()

class Cloud(Point):
    maxlimit=30
    def __init__(self):
        self.poincount=0
        self.points=[]

    def pointchech(self,point):
        if point in self.points:
            print("The point is in the cloud")
    def isFull(self):
        if self.pointcount<maxlimit:
            print("The cloud is full")
    def isEmpty(self):
        if self.poincount==0
            print("The cloud is empty")
    def add(self,point):
        self.points.append([point.x,point.y])








