import numpy as np


class ClinicalData(object):
    pass


class RSVData(ClinicalData):
    def __init__(self, N=12):
        self.N = 1 / N
        self.ages = np.arange(0, 100, 5)
        self.mu = None

