
def sumn(n):
    sum=0
    for i in range(1,n+1):
        sum=sum+i
    return sum

a=int(input("Enter the value of n:"))
print("The sum is ", sumn(a))
