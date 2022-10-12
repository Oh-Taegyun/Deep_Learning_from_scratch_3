class Variable:
    def __init__(self, data):
        self.data = data #������ ���� 
        self.grad = None #���� ����
        self.creator = None # �� ������ ������ â���ڸ� ����

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() #�Լ��� ����
            x, y = f.input, f.output 
            x.grad = f.backward(y.grad) #���� ����

            if x.creator is not None:
                funcs.append(x.creator) 


