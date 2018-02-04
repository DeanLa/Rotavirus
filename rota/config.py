from itertools import product
import numpy as np
import logging
import sys
from datetime import datetime
# Pandas print options
np.set_printoptions(precision=4, suppress=True, linewidth=80)

file_name = str(datetime.now()).replace(" ","-").replace(":","-")
# file_name = file_name[:file_name.find('.')]+'.log'
file_name = 'my_log.log'
# print (file_name)
# Logger class
logger = logging.getLogger('pertussis')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)-s, %(message)s')

sh = logging.StreamHandler()
sh.setFormatter(formatter)
sh.setLevel(logging.CRITICAL)

fh = logging.FileHandler('./log/'+file_name, mode='w')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)


# progress = logging.getLogger('progress')
# pb = logging.FileHandler('./progress.log', mode='w')
# pb.setFormatter(formatter)
# pb.setLevel(logging.DEBUG)

logger.addHandler(sh)
logger.addHandler(fh)
# progress.addHandler(pb)