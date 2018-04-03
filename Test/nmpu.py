import numpy as np
def f(x,y,z):
    return y+z
a=np.fromfunction(f,(2,3,5))
a.sum(0)
print(a)
a.dot()
print(a)
