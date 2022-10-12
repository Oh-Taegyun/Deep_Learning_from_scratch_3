import numpy as np
from Variable import Variable
from Function import Function

class Exp(Function):
    def forward(self,x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy 
        return gx


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2*x*gy
        return gx


    # 이 밑으로는 파이썬 함수로 변환
    def square(x):
        return Square()(x)

    def exp(x):
        return Exp()(x)