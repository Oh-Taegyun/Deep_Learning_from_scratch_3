from turtle import forward
import numpy as np
from Variable import Variable
from Function import Function

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

class Mul(Function):
    def forward(self,x0,x1):
        y = x0 * x1
        return

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

class Neg(Function):
    def forward(self, x):
        return -x
    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, xs):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx





# 이 밑으로는 파이썬 함수로 변환
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0,x1):
    return Add()(x0,x1)

def mul(x0,x1):
    return Mul()(x0,x1)

def neg(X):
    return Neg() (x)

def sub(x0,x1):
    x1 = as_array(x1)
    return Sub() (x0,x1)

def rsub(x0,x1):
    x1 = as_array(x1)
    return Sub() (x1,x0)

def div(x0,x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1,x0)

def pow(x,c):
    return Pow(c)(x)


#보다 간편한 계산을 위한 연산자 오버로드
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__add__ = add
Variable.__radd__ = add
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow
