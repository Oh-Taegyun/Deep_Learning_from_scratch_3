class Variable:
    def __init__(self, data):
        if data is not None: #ndarray만 취급하게끔 설정
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        
        self.data = data # 데이터 저장 
        self.grad = None # 기울기 저장
        self.creator = None # 이 변수를 저장한 창조자를 저장

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            # data와 형상과 데이터 타입이 같은 ndarray 인스턴스를 생성하는데, 모든 요소를 1로 채워서 돌려줍니다. 역전파때 1을 입력하는것을 생략하기 위함
            self.grad = np.ones_like(self.data) 
            
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 함수를 꺼냄
            x, y = f.input, f.output 
            x.grad = f.backward(y.grad) # 기울기 저장

            if x.creator is not None:
                funcs.append(x.creator) 


