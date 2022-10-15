import contextlib
from Variable import *


class Config:
    enable_backprop = True # True면 역전파 코드 활성화 False면 역전파 코드 비활성화


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

'''
#근데 try, finally로도 구현이 가능하지만 컨텍스트 매니저로 이용하면 더 깔끔할것 같다고 시작했으나, 클래스 안에서(using_config) 
타 클래스를 인스턴스화 하지 않고 클래스 그 자체로 전달하는 과정이 이상함을 발견 굳이 저 부분을 해결하려고 하면 오히려 더 깔끔하지가 않아서 포기

class using_config:
    def __call__(self, name, value):
        self.name = name
        self.value = value

    def __enter__(self):
        self.old_value = getattr(Config, self.name) #오브젝트(config)안에 찾고자 하는 변수(name)값을 출력
        setattr(Config,self.name,self.value) #오브젝트(config)안에 새로운 변수(name)를 추가하고 값은(value)로 설정

    def __exit__(self,type,value,traceback):
        setattr(Config,self.name,self.old_value)
        
def no_grad():
    return using_config()('enable_backprop', False)

'''

