def sumn(n):
    sum=0
    for i in range(1,n+1):
        sum=sum+i
    return sum

def factorial(n):
    fac=1
    for i in range(1,n+1):
        fac=fac*i
    return fac

a=int(input("Enter the value of n:"))
ssum=0
for i in range(1,a+1):
    msum=1
    for j in range(1,i+1):
        msum=msum*sumn(j)

    ssum=ssum+((-1)**(2*i-1)*msum)/factorial(i)

print(ssum)