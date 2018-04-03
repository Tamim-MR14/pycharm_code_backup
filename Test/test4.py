a=input("Give a integer number")
b=len(a)
a=int(a)
sum=0
for x in range(0,b):
    sum=sum+a%10
    a=a//10

print(sum)
