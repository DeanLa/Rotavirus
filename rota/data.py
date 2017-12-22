import numpy as np


class ClinicalData(object):
    pass


class RotaData(ClinicalData):
    def __init__(self, N=12):
        self.N = 1 / N
        self.ages = np.arange(0, 2 / 12, 4 / 12, 6 / 12, )
        self.mu = None
        self.C = np.genfromtxt('./data/contact-mixing.csv')


if __name__ == '__main__':
    d = RotaData()
    print(d.C)
