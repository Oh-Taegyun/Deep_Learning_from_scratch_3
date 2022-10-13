from turtle import forward
import numpy as np
from Variable import Variable
from Function import Function

#입력이 스칼라인 경우 ndarray 인스턴스로 변환해 주는 함수
def as_array(x):
    if np.isscalar(x): #스칼라 타입인지 확인해주는 함수
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
        x = self.inputs[0].data #가변 인자 함수로 입력값이 튜플로 인식되었기 때문에 inputs[0]이다.
        gx = 2*x*gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


# 이 밑으로는 파이썬 함수로 변환
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0,x1):
    return Add()(x0,x1)