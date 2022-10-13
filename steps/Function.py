import numpy as np
from Variable import *
from utility import *
import weakref

class Config: 
    enable_backprop = True # True�� ������ �ڵ� Ȱ��ȭ False�� ������ �ڵ� ��Ȱ��ȭ

class Function:
    def __call__(self, *inputs): #���� ���� �Լ��� ����
        xs = [x.data for x in inputs] #����Ʈ ����
        ys = self.forward(*xs)
        if not isinstance(ys, tuple): 
            ys = (ys,)
        outputs =[ Variable(as_array(y) for y in ys)] #y�� ��Į���� ��� ndarray �ν��Ͻ��� ��ȯ

        if Config.enable_backprop: # �н� �ÿ��� �̺а��� ���ؾ� �ؼ� �Է°��� �����ؾ�������, �߷нÿ��� �����ĸ� �ϱ� ������ ��� ����� ����
            self.generation = max([x.generation for x in inputs])

            for output in outputs: # ��� ������ â���� ����
                output.set_creator(self)

            self.inputs = inputs # �Է� ������ �����
            self.outputs = [weakref.ref(output) for output in outputs] #��ȯ ������ ���� ���ؼ� ���� ������ ����

        return outputs if len(outputs) > 1 else output[0]

    def forward(self, xs):
        raise NOTImplementedError()

    def backward(self, gys):
        raise NOTImplementedError()

