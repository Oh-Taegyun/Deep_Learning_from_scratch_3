from turtle import forward
import numpy as np
from Variable import Variable
from Function import Function

#�Է��� ��Į���� ��� ndarray �ν��Ͻ��� ��ȯ�� �ִ� �Լ�
def as_array(x):
    if np.isscaler(x): #��Į�� Ÿ������ Ȯ�����ִ� �Լ�
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
        x = self.inputs[0].data #���� ���� �Լ��� �Է°��� Ʃ�÷� �νĵǾ��� ������ inputs[0]�̴�.
        gx = 2*x*gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


# �� �����δ� ���̽� �Լ��� ��ȯ
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0,x1):
    return Add()(x0,x1)