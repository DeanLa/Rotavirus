# if __name__ == '__main__':
#     from rota import *
#
#     rota = Rota('chain_01')
import matplotlib.pyplot as plt
import pandas as pd

from rota import *

if __name__ == '__main__':
    logger.info("start")
    # z = Stochastic('z', 0, 1, .5)
    b1 = Stochastic('b1', 0, 2, initial=0.01)
    b2 = Stochastic('b2', 0, 2, initial=0.01)
    b3 = Stochastic('b3', 0, 2, initial=0.01)
    b4 = Stochastic('b4', 0, 2, initial=0.01)
    b5 = Stochastic('b5', 0, 2, initial=0.01)
    offset = Stochastic('offset', 0, 2 * np.pi)
    vars = [b1, b2, b3, b4, b5, offset]
    extra = {'start': 0, 'end': 10, 'scaling_factor': 0.2, 'years_prior': 10}
    x = Rota('x', vars, extra, rota_eq)
    r = rota_eq(x,)
    A = sum(r)
    A = pd.DataFrame(A.T, columns=RotaData().a_l)
    print(type(x))

    # print(Chains(12, 12, 55, 12).chains)
    # logger.info("start")
    # z = Stochastic('z', 0, 1, .5)
    # d = {'start': 0, 'end': 10, 'scaling_factor': 0.2, 'years_prior': 10}
    # x = Rota('x', [z], d, rota_eq)
    # result = rota_eq(x, RotaData())
    # print(type(x))
    # plt.plot(result.S1)
    # print(Chains(12, 12, 55, 12).chains)
