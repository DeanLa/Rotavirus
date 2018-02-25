from tqdm import trange, tqdm
import logging
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal as multinorm
from functools import partial
from rota import rota_eq, RotaData, collect_state_0, COMP, state_z_to_state_0, logger, Disease, Rota
from rota.funcs import save_mcmc, load_mcmc, log_likelihood

class Simulation(object):
    def __init__(self, mcmc: Disease, simulations: np.array):
        self.mcmc = mcmc
        self.mcmc.tally = 5000
        np.random.randint(self.mcmc.tally, len(self.mcmc)-1, size = simulations)
        self.start = self.mcmc.end
        self.end = self.start + 15
        self.eq_func = self.mcmc.eq_func
        self.resolution = self.mcmc.resolution
        self.equations = partial(self.eq_func, self)
        self.cover1 = 0.97
        self.cover2 = 1
        self.cover3 = 1

        self.simulated_names = ['b1','b2','b3','b4','b5','offset'
                                'reduct1','reduct2','reduct3',
                                'rhoav1','rhomv1','rhosv1',
                                'rhoav2','rhomv2','rhosv2',
                                'rhoav3','rhomv3','rhosv3']

        self.chain_subset = self.mcmc
    def run_model(self):
        pass

    def generate_runs(self):
        pass

    def set_stochastics(self):
        pass

    def set_quad(self):
        pass

    # def

if __name__ == '__main__':
    mcmc = Rota.load('./chains/Final.pkl')
    mcmc.eq_func = rota_eq