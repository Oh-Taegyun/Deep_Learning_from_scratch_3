import numpy as np
from Variable import *
from utility import *
from Function import *
from using_config import *

with no_grad():
    x = Variable(np.array(1.0))
    y = square(square(square(x)))
    print(y.data)


