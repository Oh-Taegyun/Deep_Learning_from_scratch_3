import numpy as np
from utility import *

class Variable:
    __array_priority__ = 200 # Variable 인스턴스의 연산자 우선순위를 ndarray 인스턴스의 연산자 우선순위보다 높이는 기능
    def __init__(self, data, name=None):
        if data is not None: # ndarray만 취급하도록 설정
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        
        self.data = data # 데이터 저장
        self.name = name # 변수에 붙일 이름
        self.grad = None # 기울기 저장
        self.creator = None # 이 변수를 저장한 창조자를 저장
        self.generation = 0  # 세대 수를 기록하는 변수 (복잡한 계산을 위한 변수)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 

    def backward(self, retain_grad = False): # 변수로부터 역전파를 실행할 수 있게끔 역전파 함수 생성
        if self.grad is None:
            # data와 형상과 데이터 타입이 같은 ndarray 인스턴스를 생성하는데, 모든 요소를 1로 채워서 돌려줍니다. 역전파때 1을 입력하는것을 생략하기 위함
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
            f = funcs.pop() # 함수를 꺼냄
            gys = [output().grad for output in f.outputs] # 참조된 데이터에 접근하려면 b()처럼 쓰면 된다.
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

            if not retain_grad: #retain_grad가 Ture면 모든 변수가 기울기를 유지 False면 중간 변수의 미분값을 모두 None
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):  #미분값을 초기화 하는 함수
        self.grad = None

    @property # 이 한줄덕분에 shape 메서드를 인스턴스 변수처럼 사용할 수 있음.
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.daat).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    '''
    def __mul__(self, other):
        return mul(self,other)
    더 간단하게 하는 방법이 있는데 Variable.__mul__ = mul로 하면 간단하다.
    '''
  


