import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None: # ndarray�� ����ϰԲ� ����
            if not isinstance(data, np.ndarray):
                raise TypeError('{}��(��) �������� �ʽ��ϴ�.'.format(type(data)))
        
        self.data = data # ������ ���� 
        self.grad = None # ���� ����
        self.creator = None # �� ������ ������ â���ڸ� ����
        self.generation = 0 # ���� ���� ����ϴ� ���� (������ ����� ���� ����)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 

    def backward(self, retain_grad = False): # �����κ��� �����ĸ� ������ �� �ְԲ� ������ �Լ� ����
        if self.grad is None:
            # data�� ����� ������ Ÿ���� ���� ndarray �ν��Ͻ��� �����ϴµ�, ��� ��Ҹ� 1�� ä���� �����ݴϴ�. �����Ķ� 1�� �Է��ϴ°��� �����ϱ� ����
            self.grad = np.ones_like(self.data) 
            
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop() # �Լ��� ����
            gys = [output().grad for output in f.outputs] # ������ �����Ϳ� �����Ϸ��� b()ó�� ���� �ȴ�.
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad: #retain_grad�� Ture�� ��� ������ ���⸦ ���� False�� �߰� ������ �̺а��� ��� None
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self): #�̺а��� �ʱ�ȭ �ϴ� �Լ�
        self.grad = None







