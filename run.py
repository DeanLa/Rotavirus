# if __name__ == '__main__':
#     from rota import *
#
#     rota = Rota('chain_01')

from rota import *

if __name__ == '__main__':
    logger.info("start")
    z = Stochastic('z', 0, 1, .5)
    d = {'start': 0, 'end': 10, 'scaling_factor': 0.2, 'years_prior': 10}
    x = Rota('x', [z], d, rota_eq)
    result = rota_eq(x, RotaData())
    print(type(x))
    plt.plot(result.S1)
    print(Chains(12, 12, 55, 12).chains)
