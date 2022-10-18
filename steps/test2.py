import numpy as np
from Dezore import *
def f(x):
    y = x ** 2
    return y

x = Variable(np.array(7))
y = f(x)
y.backward()

print (x.grad)

#나중에 이거 이용해서 자동 미분 게산기 만들어보기