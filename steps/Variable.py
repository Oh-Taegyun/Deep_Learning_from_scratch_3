class Variable:
    def __init__(self, data):
        self.data = data #데이터 저장 
        self.grad = None #기울기 저장
        self.creator = None # 이 변수를 저장한 창조자를 저장

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() #함수를 꺼냄
            x, y = f.input, f.output 
            x.grad = f.backward(y.grad) #기울기 저장

            if x.creator is not None:
                funcs.append(x.creator) 


