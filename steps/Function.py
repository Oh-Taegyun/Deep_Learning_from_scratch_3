import numpy as np
from Variable import *
from utility import *

class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)

        self.input = input # 입력 변수를 기억함
        self.output = output
        return output

    def forward(self, x):
        raise NOTImplementedError()

    def backward(self, gy):
        raise NOTImplementedError()


