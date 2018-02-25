from collections import namedtuple

COMP = namedtuple("Compartments",
                  '''M S1 Ia1 Im1 Is1 R1 S2 Ia2 Im2 Is2 R2 S3 Ia3 Im3 Is3 R3 
                  V1 V2 V3 Iav Imv Isv''')

# INITIAL
from .config import *
from .funcs import *
from .stopwatch import *
# MODEL
# from .parameters import *
from .data import *
from .equations import *
from .model import * # Requires Parameters
from .simulation import *
# # POST
from .charts import *
# from .report import *
# from .analysis import *