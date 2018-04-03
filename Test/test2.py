x  = [int(x) for x in input("Enter the coordinates here: ").split()]
if (x[3]-x[1])/(x[2]-x[0])==(x[7]-x[5])/(x[6]-x[4]):
    print("The two lines are parallel")
else:
    print("The two lines are intersecting")
