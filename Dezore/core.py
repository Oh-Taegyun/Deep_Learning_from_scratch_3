class Config:
    enable_backprop = True # True면 역전파 코드 활성화 False면 역전파 코드 비활성화


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

    def backward(self, retain_grad=False):  # 변수로부터 역전파를 실행할 수 있게끔 역전파 함수 생성
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
            f = funcs.pop()  # 함수를 꺼냄
            gys = [output().grad for output in f.outputs]  # 참조된 데이터에 접근하려면 b()처럼 쓰면 된다.
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
        p = str(self.daat).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'


class Function:
    def __call__(self, *inputs): #가변 인자 함수로 받음
        inputs = [as_varialbe(x) for x in inputs]
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
        raise NOTImplementedError()

    def backward(self, gys):
        raise NOTImplementedError()


def as_array(x):
    if np.isscalar(x):  # 스칼라 타입인지 확인해주는 함수
        return np.array(x)
    return x


class Exp(Function):
    def forward(self, x):
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
        x = self.inputs[0].data  # 가변 인자 함수로 입력값이 튜플로 인식되었기 때문에 inputs[0]이다.
        gx = 2 * x * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class Mul(Function):
    def forward(self, x0, x1):
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
        gx = c * x ** (c - 1) * gy
        return gx


# 이 밑으로는 파이썬 함수로 변환
def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def add(x0, x1):
    return Add()(x0, x1)


def mul(x0, x1):
    return Mul()(x0, x1)


def neg(X):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


def pow(x, c):
    return Pow(c)(x)


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







