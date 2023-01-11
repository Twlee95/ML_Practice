X=list(range(10))
Y=[1,1,2,4,5,7,8,9,9,10]
import matplotlib.pyplot as plt
plt.plot(X,Y)

class H():

    def __init__(self,w):
        self.w = w

    def forward(self, x):
        return self.w * x

def cost(h, X, Y):
    error = 0
    for i in range(len(X)):
        error += (h.forward(X[i])-Y[i])**2
    error /= len(X)
    return error

h = H(1)

C= cost(h,X,Y)

def cal_grad(w):
    h=H(w)
    cost1= cost(h,X,Y)
    eps =0.001
    h= H(w+eps)
    cost2 = cost(h,X,Y)
    dcost = cost2 - cost1
    dw = eps
    grad = dcost/dw
    return grad

a= cal_grad(4)
w=4
lr = 0.01
for i in range(10):
   w = w + lr*(-cal_grad(w))
   print(w)