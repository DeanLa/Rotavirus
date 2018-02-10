
import matplotlib.pyplot as plt
import os

import multiprocessing
import pandas as pd

# from funcs import log_likelihood
from rota import *


def run_multi_random(name, subdir='', runs=100):
    print ("Starting Chain {}".format(name))
    b1 = Stochastic('b1', 0, 4)
    b2 = Stochastic('b2', 0, 6)
    b3 = Stochastic('b3', 0, 2.5)
    b4 = Stochastic('b4', 0, 1)
    b5 = Stochastic('b5', 0, 0.5)
    A = Stochastic('A', 0, 10, initial=5)
    offset = Stochastic('offset', 0, 10, cyclic=True)
    vars = [b1, b2, b3, b4, b5, offset]
    model = Model(vars)

    extra = {'start': 0, 'end': 9, 'scaling_factor': 0.2, 'years_prior': 10,
             'resolution': 4, 'save_path':'./random_search/{}/'.format(subdir)}
    try:
        os.mkdir('./random_search/{}'.format(subdir))
    except FileExistsError:
        print ('folder exists will overwrite')
    mcmc = Rota('mcmc_mp_{}'.format(name), model, extra, rota_eq)
    mcmc.save()
    for run in range(runs):
        mcmc.random_run()
    return mcmc

def run_multi_sample(name, subdir='', runs=10000):
    print ("Starting Chain {}".format(name))
    b1 = Stochastic('b1', 0, 5)
    b2 = Stochastic('b2', 0, 10)
    b3 = Stochastic('b3', 0, 10)
    b4 = Stochastic('b4', 0, 10)
    b5 = Stochastic('b5', 0, 10)
    offset = Stochastic('offset', 0, 10, cyclic=True)
    vars = [b1, b2, b3, b4, b5, offset]
    model = Model(vars)

    extra = {'start': 0, 'end': 9, 'scaling_factor': 0.2, 'years_prior': 50,
             'resolution': 4, 'save_path':'./chains/{}/'.format(subdir)}
    try:
        os.mkdir('./chains/{}'.format(subdir))
    except FileExistsError:
        print ('folder exists will overwrite')
    mcmc = Rota('mcmc_mp_{}'.format(name), model, extra, rota_eq)
    mcmc.save()
    # for run in range(runs):
    mcmc.sample(runs,500)
    return mcmc


if __name__ == '__main__':
    n = 16
    subdir = 'c0209'
    runs = 15000
    pool = multiprocessing.Pool(n)
    pool.starmap(run_multi_sample, zip([str(proc) for proc in range(n)],
                                      n*[subdir],
                                      n*[runs]))
    pool.close()
    pool.join()


    # logger.info("start")
    # mcmc = run_multi('a1','d',runs=1000)
    # plot_against_data(mcmc.yhat_history[1],mcmc.ydata); plt.show()
