from Dezore import Layer
from Dezore import utils
import Dezore.functions as F
import Dezore.layers as L

class Model(Layer): #그림 그리는 기능 추가
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file) #verbose 가 True면 ndarray 인스턴스의 형상과 타입도 계산 그래프에 표시해줌

class MLP(Model):
    def __init__(self, fc_output_sizes, activation = F.sigmold):
        # fc_output_sizes는 신경망을 구성하는 완전연결계층들의 출력 크기를 튜플 또는 리스트로 지정합니다. 
        # activation은 활성화 함수를 지정합니다
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self,'l' + str(i), layer) #l1 = L.Linear(out_size) 처럼 객체에 변수 저장중 
            self.layers.append(layer)

    def forward(self,x):
        for i  in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)