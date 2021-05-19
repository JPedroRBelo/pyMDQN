'''import random
L = [(3,4,5),(1,4,5),(1,2,3),(1,2,3)]

t1 = [L.pop(random.randrange(len(L))) for _ in range(2)]
print(t1)

from shutil import copy

import Net as net
print(net.__file__	)
copy("testsample.py","datasets/" )
'''




import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO) # process everything, even if everything isn't printed

ch = logging.StreamHandler()
ch.setLevel(logging.INFO) # or any other level
logger.addHandler(ch)


fh = logging.FileHandler('myLog.log')
fh.setLevel(logging.INFO) # or any level you want
logger.addHandler(fh)


# print(foo)
logger.info('testing')

# print('finishing processing')
logger.info('finishing processing')

# print('Something may be wrong')
logger.warning('Something may be wrong')

# print('Something is going really bad')
logger.error('Something is going really bad')
