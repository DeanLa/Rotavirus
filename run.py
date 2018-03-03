# if __name__ == '__main__':
#     from rota import *
#
#     rota = Rota('chain_01')
import matplotlib.pyplot as plt
import pandas as pd

# from funcs import log_likelihood
from rota import *

if __name__ == '__main__':
    logger.info("start")
    new_mcmc = True
    if new_mcmc:
        b1 = Stochastic('b1', 0, 10, initial=3 )
        b2 = Stochastic('b2', 0, 10, initial=5 * 1.5)
        b3 = Stochastic('b3', 0, 10, initial=1.6 * 1.5)
        b4 = Stochastic('b4', 0, 10, initial=1 * 1.5)
        b5 = Stochastic('b5', 0, 10, initial=0.336 * 1.5)
        # A = Stochastic('A', 0, 10, initial=5)

        offset = Stochastic('offset', 0, 10, initial=3, cyclic=True)

        vars = [b1, b2, b3, b4, b5, offset]
        model = Model(vars)
        inc = 1.06
        vals = list(np.array([ 2.0436,  3.8853,  1.2249,  0.718 ,  0.1172,  2.5374]) * inc)
        model.update(vals)
        # model.update([0.0000003, 0.0000005, 0.00000016, 4e-7, 3.36e-8, 1])
        extra = {'start': 0, 'end': 9, 'scaling_factor': 0.2, 'years_prior': 10,
                 'resolution': 4}
        with PrintElapsedTime("init mcmc"):
            mcmc = Rota('mcmc', model, extra, rota_eq, )
    else:
        mcmc = Rota.load('./mcmc180208.pkl')
    # print(log_likelihood(x.y_now, x.ydata, x.sigma))
    # fig, ax = plot_against_data(x.y_now, x.ydata)
    # fig.suptitle('{}  --- {}'.format(extra['resolution'], log_likelihood(x.y_now, x.ydata, x.sigma)))
    # plt.show()
    best = mcmc.yhat_history[0]
    fig, ax = plot_against_data(best, mcmc.ydata_cases)
    fig, ax = plot_difference(best, mcmc.ydata)
    # ll = log_likelihood(best,mcmc.ydata,mcmc.sigma)
    fig.suptitle('{:.0f}      {}'.format(ll, inc))
    print(ll)
    plt.show()

    with PrintElapsedTime():
        mcmc.sample(10000, 250)

    x=1
    # r = list(r)
    # rU = COMP._make([np.hstack((a, b, c)) for a, b,c  in zip(r[0], r[1], r[2])])
    # r = x.equations(steps=4 * 52, start=-10)
    # rU = r
    # plot_compartments(rU, 'Is1 Ia1')
    # plt.show()
    # A = sum(rU)
    # A = pd.DataFrame(A.T, columns=RotaData().a_l)
    # plt.plot(A)
    # plt.show()
    # plt.plot(A.sum(axis=1))
    # plt.show()
    # print(type(x))

    # print(Chains(12, 12, 55, 12).chains)
    # logger.info("start")
    # z = Stochastic('z', 0, 1, .5)
    # d = {'start': 0, 'end': 10, 'scaling_factor': 0.2, 'years_prior': 10}
    # x = Rota('x', [z], d, rota_eq)
    # result = rota_eq(x, RotaData())
    # print(type(x))
    # plt.plot(result.S1)
    # print(Chains(12, 12, 55, 12).chains)
