import numpy as np
from Variable import *
from utility import *
from using_config import *
import weakref


#입력이 스칼라인 경우 ndarray 인스턴스로 변환해 주는 함수
def as_array(x):
    if np.isscalar(x): #스칼라 타입인지 확인해주는 함수
        return np.array(x)
    return x

# obj가 variable 인스턴스가 아닐 경우 변환해서 반환하는 기능
def as_varialbe(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

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

