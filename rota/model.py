from scipy.stats import uniform
from tqdm import trange, tqdm
import logging
import numpy as np
from rota import *
from rota import rota_eq

logger = logging.getLogger(__name__)
class Stochastic(object):
    def __init__(self, name, a, b, initial=None):
        self.name = name
        self.a = a
        self.b = b
        self.dist = uniform(a, b - a)
        self.value = initial if initial else self.dist.rvs()

    def check_proposal(self):
        return self.a < self.value < self.b

    def update(self, value):
        self.prev = self.value
        self.value = value

    def rollback(self):
        self.value = self.prev

    def __str__(self):
        if self.check_proposal():
            return (f"{self.name} = {self.value} |  in {self.a}[---]{self.b}")
        else:
            return (f"{self.name} = {self.value} |  NOT in {self.a}[---]{self.b}")

    def __repr__(self):
        return self.__str__()

class Disease(object):
    def __init__(self, name, stochastics, **kwargs):
        self.active = True
        self.yshape = None
        self.xshape = None
        self.get_data()
        self.name = name
        self.stochastics = stochastics

        # Constants
        self.accept_hat = 0.23
        self.sigma = 1
        self.state_0 = None
        self.start = None
        self.years_prior = None
        self.end = None
        self.fixed = None
        # Initial Values
        self.values = [s.value for s in self.stochastics]
        self.initial_values = [s.value for s in self.stochastics]
        self.names = [s.name for s in self.stochastics]
        self.d = len(self.values)
        # self.y_hat, self.state_z = self.run_model(self)

        # Chains
        self.yhat_history = np.zeros(np.append(0, self.yshape))
        self.state_z_history = np.zeros(np.append(0, self.yshape))
        self.chain = np.zeros((0, self.d))
        self.guesses = np.zeros((0, self.d))
        # Metrics
        self.accepted = np.ones(0)
        self.rates = np.ones((0))
        self.mle = -np.inf
        self.gelman_rubin = np.zeros((0,self.d))
        self.change = np.ones((0))
        self.ll_history = np.ones((0,2))
        # Stochastics
        self.get_stochastics(self.stochastics)
        #
        self.populate(kwargs)
        self.compute_jump()

    def get_stochastics(self, stochastics):
        for var in stochastics:
            setattr(self, var.name, var.value)

    def compute_jump(self, scaling_stop_after = 10000, sd_stop_after = 10000):
        # Model Specific
        self.scaling_factor = np.array([2.4 / np.sqrt(self.d)])
        self.cov = np.diag(np.ones(self.d))
        self.sd_stop_after = sd_stop_after
        self.scaling_stop_after = scaling_stop_after

        # After updating
        self.sd = self.scaling_factor ** 2 * self.cov

    def update(self):
        '''updates the chains, rates, etc'''

    def populate(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def clinical_data(self, data):
        '''Takes a ClinicalData object of constants and populates them to the model'''

        # Do something with dicts or json...
        # Use populate to populate them

    def sample(self, recalculate=500):
        compute_scaling_factor = self.scaling_stop_after > len(self)
        compute_sd = self.sd_stop_after < len(self)
        for iteration in trange(recalculate, desc=self.name, leave=False, position=0):
            if not self.active: continue


    def run_model(self):
        '''returns the results of the model and the state at last point of time'''
        raise NotImplementedError

    def run_equations(self):
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.chain)

    def __str__(self):
        return (f"{self.name}\n{self.names}\n{self.values}")
    __repr__ = __str__

class Rota(Disease):

    def __init__(self, name, stochastics, populate_values, eq_func):
        super(Rota, self).__init__(name, stochastics)
        self.stochastics = stochastics
        self.equations = eq_func

        # self.get_data()
        fixed = RotaData()
        self.state_0 = collect_state_0(fixed)
        self.populate(populate_values)

    # def run_equations(self):
    #     self.equations()

    def run_model(self):
        self.equations(self)

    def get_data(self):
        self.xdata = np.arange(5)
        self.ydata = np.vstack((self.xdata ** 2, self.xdata))
        self.xshape = self.xdata.shape
        self.yshape = self.ydata.shape
        # assert sim1 of x is like y


class Chains(object):
    def __init__(self, *args):
        self.chains = list(args)





if __name__ == '__main__':
    logger.info("start")
    # z = Stochastic('z', 0, 1, .5)
    b1 = Stochastic('b1', 0, 2)
    b2 = Stochastic('b2', 0, 2)
    b3 = Stochastic('b3', 0, 2)
    b4 = Stochastic('b4', 0, 2)
    b5 = Stochastic('b5', 0, 2)
    phi = Stochastic('phi', 0, 2 * np.pi)
    vars = [b1,b2,b3,b4,b5,phi]
    extra = {'t':5,'scaling_factor':0.2}
    x = Rota('x', vars, extra, rota_eq)
    rota_eq(x)
    print(type(x))
    print(Chains(12, 12, 55, 12).chains)
