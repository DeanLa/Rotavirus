# if __name__ == '__main__':
#     from rota import *
#
#     rota = Rota('chain_01')
import matplotlib.pyplot as plt
import pandas as pd

from rota import *

if __name__ == '__main__':
    logger.info("start")
    b1 = Stochastic('b1', 0, 2, initial=0.0001)
    b2 = Stochastic('b2', 0, 2, initial=0.0001)
    b3 = Stochastic('b3', 0, 2, initial=0.0001)
    b4 = Stochastic('b4', 0, 2, initial=0.0001)
    b5 = Stochastic('b5', 0, 2, initial=0.0001)
    offset = Stochastic('offset', 0, 2 * np.pi)
    vars = [b1, b2, b3, b4, b5, offset]
    extra = {'start': 0, 'end': 10, 'scaling_factor': 0.2, 'years_prior': 10}
    x = Rota('x', vars, extra, rota_eq)
    r = x.run_equations()
    A = sum(r)
    A = pd.DataFrame(A.T, columns=RotaData().a_l)
    plt.plot(A)
    plt.show()
    plt.plot(A.sum(axis=1))
    plt.show()
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
