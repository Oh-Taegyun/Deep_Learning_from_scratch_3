import numpy as np
import contextlib
import weakref

import Dezore.functions


# ----------------------------------------------------------------------------------
# Config - True면 역전파 코드 활성화 False면 역전파 코드 비활성화
# ----------------------------------------------------------------------------------
class Config:
    enable_backprop = True # True면 역전파 코드 활성화 False면 역전파 코드 비활성화

#입력이 스칼라인 경우 ndarray 인스턴스로 변환해 주는 함수



# ----------------------------------------------------------------------------------
# 다양한 함수들
# ----------------------------------------------------------------------------------
def as_array(x):
    if np.isscalar(x): #스칼라 타입인지 확인해주는 함수
        return np.array(x)
    return x

# obj가 variable 인스턴스가 아닐 경우 변환해서 반환하는 기능
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config,name) #오브젝트(config)안에 찾고자 하는 변수(name)값을 출력
    setattr(Config, name, value) #오브젝트(config)안에 새로운 변수(name)를 추가하고 값은(value)로 설정
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)


# ----------------------------------------------------------------------------------
# Variable / Function
# ----------------------------------------------------------------------------------
class Variable:
    __array_priority__ = 200  # Variable 인스턴스의 연산자 우선순위를 ndarray 인스턴스의 연산자 우선순위보다 높이는 기능

    def __init__(self, data, name=None):
        if data is not None:  # ndarray만 취급하도록 설정
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data  # 데이터 저장
        self.name = name  # 변수에 붙일 이름
        self.grad = None  # 기울기 저장
        self.creator = None  # 이 변수를 저장한 창조자를 저장
        self.generation = 0  # 세대 수를 기록하는 변수 (복잡한 계산을 위한 변수)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):  # 변수로부터 역전파를 실행할 수 있게끔 역전파 함수 생성
        if self.grad is None:
            # data와 형상과 데이터 타입이 같은 ndarray 인스턴스를 생성하는데, 모든 요소를 1로 채워서 돌려줍니다. 역전파때 1을 입력하는것을 생략하기 위함
            # self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data)) # 고차 미분을 위한 구현

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()  # 함수를 꺼냄
            gys = [output().grad for output in f.outputs]  # 참조된 데이터에 접근하려면 b()처럼 쓰면 된다.

            with using_config('enable_backprop',create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx # 만일 기울기가 있다면 누적

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:  # retain_grad가 Ture면 모든 변수가 기울기를 유지 False면 중간 변수의 미분값을 모두 None
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):  # 미분값을 초기화 하는 함수
        self.grad = None

    @property  # 이 한줄덕분에 shape 메서드를 인스턴스 변수처럼 사용할 수 있음.
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
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def reshape(self, *shape): # 데이터 형상 바꾸는 함수
        if len(shape) == 1 and isinstance(shape[0],(tuple, list)):
            shape = shape[0]
        return Dezore.functions.reshape(self, shape)

    def traspose(self, *axes): # 전치 행렬을 구하기 위한 함수
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return Dezore.functions.transpose(self, axes)

    @property
    def T(self):
        return Dezore.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return Dezore.functions.sum(self, axis,keepdims)


class Function:
    def __call__(self, *inputs): #가변 인자 함수로 받음
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs] #리스트 내포
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs =[Variable(as_array(y)) for y in ys] #y가 스칼라인 경우 ndarray 인스턴스로 변환

        if Config.enable_backprop: # 학습 시에는 미분값을 구해야 해서 입력값을 저장해야하지만, 추론시에는 순전파만 하기 때문에 계산 결과를 버림
            self.generation = max([x.generation for x in inputs])

            for output in outputs: # 출력 변수들 창조자 설정
                output.set_creator(self)

            self.inputs = inputs # 입력 변수를 기억함
            self.outputs = [weakref.ref(output) for output in outputs] #순환 참조를 막기 위해서 약한 참조를 만듦

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

# ----------------------------------------------------------------------------------
# 사칙연산, 연산자 오버로드
# ----------------------------------------------------------------------------------

# Add ------------------------------------------------------------------------------
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = Dezore.functions.sum_to(gx0, self.x0_shape)
            gx1 = Dezore.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
    
def add(x0, x1):
    x1 = as_array(x1) # 이제 x + 3.0 같은 계산도 가능함 as_array가 자동 형변환 해주기 때문
    return Add()(x0, x1)
# ----------------------------------------------------------------------------------

# Mul ------------------------------------------------------------------------------
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data 이 부분은 직접적으로 ndarray를 사용할 때 씀
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            gx0 = Dezore.functions.sum_to(gx0, x0.shape)
            gx1 = Dezore.functions.sum_to(gx1, x1.shape)
        return gx0, gx1
    
def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)
# ----------------------------------------------------------------------------------

# Neg ------------------------------------------------------------------------------
class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)
# ----------------------------------------------------------------------------------

# Sub ------------------------------------------------------------------------------
class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:
            gx0 = Dezore.functions.sum_to(gx0, self.x0_shape)
            gx1 = Dezore.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)
# ----------------------------------------------------------------------------------

# Div ------------------------------------------------------------------------------
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:
            gx0 = Dezore.functions.sum_to(gx0, self.x0.shape)
            gx1 = Dezore.functions.sum_to(gx1, self.x1.shape)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

# ----------------------------------------------------------------------------------

# Pow ------------------------------------------------------------------------------
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)
# ----------------------------------------------------------------------------------


# 보다 간편한 계산을 위한 연산자 오버로드
def setup_variable():
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


# ----------------------------------------------------------------------------------
# Parameter
# ----------------------------------------------------------------------------------
class Parameter(Variable): #파라미터를 담을 수 있게 하는 상자
    pass





