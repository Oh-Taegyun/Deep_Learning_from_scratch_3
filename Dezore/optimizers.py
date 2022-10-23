import numpy as np

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target): #매개변수를 갖는 클래스(Model 또는 Layer)를 인스턴스 변수인 target으로 설정합니다
        self.target = target
        return self

    def updata(self):
        #None 이외의 매개변수를 리스트에 모아줌
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param): # 구체적인 매개변수 갱신을 위한 함수
        raise NotImplementedError()

    def add_hook(self, f): #전처리를 수행하는 함수
        self.hooks.append(f)

# ----------------------------------------------------------------------------------
# SGD
# ----------------------------------------------------------------------------------

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr = 0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v