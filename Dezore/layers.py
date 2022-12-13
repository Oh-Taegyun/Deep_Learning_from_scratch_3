from Dezore.core import Parameter
import numpy as np
import Dezore.functions as F
import weakref

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value): # object에 존재하는 속성의 값을 바꾸거나 새로운 속성을 생성할때마다 호출되는 특수 메서드
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self,inputs):
        raise NotImplementedError()

    def params(self): # 파라미터를 반환함
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self): # 파라미터 초기화
        for param in self.params():
            param.cleargrad()

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None: # 만일 in_size가 지정되어 있지 않다면 나중으로 연기
            self._init_W()
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name = 'b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random(I,O).astype(self.dtype) * np.sprt(1/I)
        self.W.data = W_data

    def forward(self,x): 
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y

'''
왜 계층에는 forward만 있나요? backward도 있어야 하지 않나요?
라고 할 수 있는데

잘 보면 Linear계층 안에linear가 있다. 즉, 저 linear 함수에 의해서 계산 그래프가 형성이 된다.

계층은 파라미터 + 함수 를 포함하고 있는 컨테이너라고 생각하면 된다.

Function를 상속한 다양한 함수자(functions)에 의해서 계산 그래프가 형성될테니 계층의 backward는 함수자에 의해서 처리가 된 것이다. 
'''

