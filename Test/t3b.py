def factorial(n):
    fac=1
    for i in range(1,n+1):
        fac=fac*i
    return fac

a=int(input("Enter the value of n:"))

print("The factorial is ", factorial(a))