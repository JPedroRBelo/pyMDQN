import random
L = [(3,4,5),(1,4,5),(1,2,3),(1,2,3)]

t1 = [L.pop(random.randrange(len(L))) for _ in range(2)]
print(t1)

from shutil import copyfile

import Net as net
print(net.__file__)
copyfile("testsample.py","1.py" )