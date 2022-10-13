'''
Variable ver.1
��°� �Է��� ���� 1����� �����̾���

class Variable:
    def __init__(self, data):
        if data is not None: #ndarray�� ����ϰԲ� ����
            if not isinstance(data, np.ndarray):
                raise TypeError('{}��(��) �������� �ʽ��ϴ�.'.format(type(data)))
        
        self.data = data # ������ ���� 
        self.grad = None # ���� ����
        self.creator = None # �� ������ ������ â���ڸ� ����

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            # data�� ����� ������ Ÿ���� ���� ndarray �ν��Ͻ��� �����ϴµ�, ��� ��Ҹ� 1�� ä���� �����ݴϴ�. �����Ķ� 1�� �Է��ϴ°��� �����ϱ� ����
            self.grad = np.ones_like(self.data) 
            
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # �Լ��� ����
            x, y = f.input, f.output 
            x.grad = f.backward(y.grad) # ���� ����

            if x.creator is not None:
                funcs.append(x.creator) 

'''

'''
Function ver.1
��°� �Է��� ���� 1����� �����̾���

class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)

        self.input = input # �Է� ������ �����
        self.output = output
        return output

    def forward(self, x):
        raise NOTImplementedError()

    def backward(self, gy):
        raise NOTImplementedError()
'''

