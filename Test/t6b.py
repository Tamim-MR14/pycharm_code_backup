class Square:
    def __init__(self,a,x,y):
        self.arm=a
        self.x=x
        self.y=y

    def print(self):
        print(self.arm)
        print(self.x,end=',')
        print(self.y)
    def insideOn(self,x1,y1):
        if (x1<self.x) or (x1>(self.x+self.arm)) or (y1<self.y) or y1>(self.y+self.arm):
            return False
        else:
            return True

a=Square(5,0,0)
print(a.insideOn(1,1))