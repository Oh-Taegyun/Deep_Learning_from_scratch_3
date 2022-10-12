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


