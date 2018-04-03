s=input("Enter the string:")
a=int(input("Enter the value of n:"))


for i in range(len(s)):
    print(s[i+a-len(s)],end='')
