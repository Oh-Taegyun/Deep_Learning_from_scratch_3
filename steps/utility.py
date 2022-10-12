import numpy as np
from Variable import Variable
from Function import Function

#입력이 스칼라인 경우 ndarray 인스턴스로 변환해 주는 함수
def as_array(x):
    if np.isscaler(x): #스칼라 타입인지 확인해주는 함수
        return np.array(x)
    return x 



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