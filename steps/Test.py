import numpy as np
from Variable import *
from utility import *
from Function import *


Config.enable_backprop = True
x = Variable(np.ones((100,100,100)))
y = square(square(square(x)))
a = y.backward()

print(a)