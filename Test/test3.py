x  = [int(x) for x in input("Enter the coordinates and radius of the circle here: ").split()]
print((((x[2]-x[0])**2+((x[3]-x[1])**2))**0.5))
if (((x[2]-x[0])**2+((x[3]-x[1])**2))**0.5)==x[4]:
    print("The poin is on the circle")
elif (((x[2]-x[0])**2+((x[3]-x[1])**2))**0.5)>x[4]:
    print("The point is outside the circle")
else:
    print("The point is inside the circle")