from tqdm import trange, tqdm
import logging
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal as multinorm
from functools import partial
from rota import rota_eq, RotaData, collect_state_0, COMP, state_z_to_state_0, logger, Disease
from rota.funcs import save_mcmc, load_mcmc, log_likelihood

class Simulation(object):
    def __init__(self, mcmc: Disease):
        mcmc.tally = 5000
        np.random.randint(mcmc.tally, len(mcmc)-1, size = 1000)