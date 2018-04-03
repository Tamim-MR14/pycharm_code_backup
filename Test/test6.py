'''a=input("Enter a binary number:")
b=len(a)
bin=0
for x in range(0,b):
    bin=bin+int(a[x])*(2**(b-x-1))


print(bin)'''
'''def even(x):
    if x%2==0:
        print("even")
    else:
        print("odd")
    return

even(4)'''
'''def id(name, *t):
    print(name, end=' ')
    if len(t)>0:
        print(t[0],t[1])
    return

id("tamim",'42','3.7')'''

'''class Circle:
    def __init__(self,x,y,r):
        self.x=x
        self.y=y
        self.r=r
    def area(self):

        return 3.14*(self.r**2)
    def n( self, x1, y1):

        if ((self.x-x1)**2+(self.y-y1)**2)**0.5<self.r:
            return True
        else:
            return False
    def touch (self,c):
        if (((self.x-c.x)**2+(self.y-c.y)**2)**0.5)==(self.r+c.r):
            b="They touch each other"
            return b
        elif (((self.x-c.x)**2+(self.y-c.y)**2)**0.5)<(self.r+c.r):
            b="They intercect each other"
            return b
        else:
            b="They don't intersect each other"
            return b

c1=Circle(0,0,5)
c2=Circle(2,0,5)
print(c1.touch(c2)'''
'''l=[2,3,5]
import random
ls=[random.randrange(1,100,1) for i in range(10)]
b=[1, 51, 30, 60, 100, 15]
print(len([i for i in b if i>50]))
print(ls)
print([ls.count(i) for i in range(1,100)])'''
'''fo=open("foo.txt","w")
fo.write("I am tamim")
fo.close'''
'''fo = open("foo.txt", "r+")
str=fo.readline()
str=str+fo.readline()
str=str+fo.readline()
str=str+fo.readline()
print("read string is: ", str)'''
'''fo=open("foo.txt","r+")
x=25
while x!=0
    x=fo.readline()'''
'''fo=open("ct.txt","r+")
x='  '
x=fo.readline()
while True:
    x = fo.readline()
    if x=='': break
    x1=x.split(",")
    print(x1)
    a=int(x1[1])
    b=int(x1[2])
    print("Name %s: Max %d"%(x[0],max(a,b)))'''

'''list=[1,2,"Python",'a']
print(list)

tupple=(1,2,"print",3.14)
print(tupple)'''
'''def serial():
    i=0
    while True:
        yield i
        i+=1
obj=serial()
for i in range(10):
    print(next(obj),end='')
print()'''

'''class Employee:
    empCount=0

    def __init__(self, name, salary):
        self.name=name
        self.salary=salary
        Employee.empCount+=1

    def displaycount(self):
        print("The total number of employee  is %d" % Employee.empCount)

    def displayemployee(self):
        print("Name :", self.name, "Salary :", self.salary)

emp1=Employee("Zara", 2000)
emp2=Employee("Tasnim", 3000)
emp1.displayemployee()
emp2.displayemployee()
print("The number of total Employee is :", Employee.empCount)

print(hasattr(emp1,'salary'))
print(getattr(emp1,'salary'))
setattr(emp1, 'salary', 7000)
print(getattr(emp1,'salary'))
delattr(emp1,'salary')'''

'''class Myemp:
    salary=3000

    def __init__(self,salary):
        self.salary=salary

    def prsalary(self):
        print("The salary of the employee is:", self. salary)

emp=Myemp(40000)
print(Myemp.salary)
emp.prsalary()'''

'''class Polygon:
    def __init__(self):
        self.m=0
        print("In Polygon Constructor")
    def intro(self):
        print("I am a Polygon")

class Triangle(Polygon):
    def __init__(self):
        super(). __init__()
        print("In triangle Constructor")

    def intro(self):
        print("I am a Triangle")

t=Triangle()
t.intro()
print(t.m)'''

'''import support
support.print_func("World")'''

'''def f():
    global A
    A=5

A=1
print(A)
f()
print(A)'''

fo=open('foo.txt',"wb")
print("File Name: ", fo.name )
print("Closed or not: ", fo.closed)
print("Opening Mode: ", fo.mode)
fo.close()
print("Close or not: ", fo.closed)



