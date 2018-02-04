from tqdm import trange, tqdm
import logging
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal as multinorm
from functools import partial
from rota import rota_eq, RotaData, collect_state_0, COMP, state_z_to_state_0, logger
from rota.funcs import save_mcmc, load_mcmc, log_likelihood


# logger = logging.getLogger(__name__)


class Stochastic(object):
    def __init__(self, name, a, b, cyclic=False, initial=None):
        self.name = name
        self.a = a
        self.b = b
        self.dist = uniform(a, b - a)
        self.value = initial if initial else self.dist.rvs()
        self.cyclic = cyclic

    def check_proposal(self):
        return self.a <= self.value <= self.b

    def update(self, value):
        self.prev = self.value
        if self.cyclic:
            value %= self.b
        self.value = value

    def rollback(self):
        self.value = self.prev

    def random(self):
        value = self.dist.rvs()
        self.update(value)

    def __str__(self):
        if self.check_proposal():
            return (f"{self.name} = {self.value} |  in {self.a}[---]{self.b}")
        else:
            return (f"{self.name} = {self.value} |  NOT in {self.a}[---]{self.b}")

    def __repr__(self):
        return self.__str__()


class Model(object):
    def __init__(self, stochastics):
        self.stochastics = stochastics

    @property
    def values(self):
        return [stoch.value for stoch in self.stochastics]

    @property
    def names(self):
        return [stoch.name for stoch in self.stochastics]

    def update(self, guess):
        for i, stoch in enumerate(self.stochastics):
            stoch.update(guess[i])

    def check_proposal(self):
        return all([stoch.check_proposal() for stoch in self.stochastics])

    def rollback(self):
        for stoch in self.stochastics:
            stoch.rollback()

    def random(self):
        for stoch in self.stochastics:
            stoch.random()

    def __str__(self):
        return '\n'.join([stoch.__str__() for stoch in self.stochastics])

    def __repr__(self):
        return self.__str__()


class Disease(object):
    def __init__(self, name, model, populate_values, eq_func):
        self.name = name
        self.active = True
        self.xdata = None
        self.ydata = None
        self.yshape = None
        self.xshape = None
        self.get_data()
        # self.fixed_data = None
        self.model = model
        self.set_stochastics()
        # Constants
        self.accept_hat = 0.23
        self.sigma = 1
        self.state_0 = None
        self.start = None
        self.years_prior = None
        self.end = None
        self.fixed = None
        # Initial Values
        self.values = self.model.values
        self.initial_values = self.model.values
        self.names = model.names
        self.d = len(self.values)
        self.equations = partial(eq_func, self)
        self.populate(populate_values)
        self.y_now, self.state_z = self.run_model(resolution=6)

        # Chains
        # self.yhat_history = np.zeros(np.append(0, self.yshape))
        self.yhat_history = self.y_now[None, :, :]
        self.state_z_history = [self.state_z]
        self.chain = np.zeros((0, self.d))
        self.guesses = np.zeros((0, self.d))
        # Metrics
        self.accepted = np.ones(0)
        self.rates = np.ones((0))
        self.mle = -np.inf
        self.gelman_rubin = np.zeros((0, self.d))
        self.change = np.ones((0))
        self.ll_history = np.ones((0, 2))
        # Stochastics
        self.compute_jump()

    def set_stochastics(self):
        for stoch in self.model.stochastics:
            setattr(self, stoch.name, stoch.value)

    def compute_jump(self, scaling_stop_after=10000, sd_stop_after=10000):
        # # Model Specific
        # self.scaling_factor = np.array([2.4 / np.sqrt(self.d)])
        # self.cov = np.diag(np.ones(self.d))
        # self.sd_stop_after = sd_stop_after
        # self.scaling_stop_after = scaling_stop_after
        # After updating
        # self.sd = self.scaling_factor ** 2 * self.cov
        raise NotImplementedError

    def update(self):
        '''updates the chains, rates, etc'''

    def populate(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def random_run(self):
        logger.info('='*80)
        self.model.random()
        self.set_stochastics()
        try:
            logger.info('guess: {}'.format(self.model.values))
            y_star, _ = self.run_model(7)
            ll_star = log_likelihood(y_star, self.ydata, self.sigma)
            logger.info('LL:    {}'.format(ll_star))
            # if ll_star < -(1e20):
            #     logger.warning('bad set for model {} with guess {}'.format(self.name, iteration))
        except Exception as e:
            logger.error('exception at model {} PROBABLY S-I-R fail'.format(self.name))
            # y_star, state_z = self.no_likelihood
            # ll_star = -np.inf
    def sample(self, recalculate=500):
        compute_scaling_factor = self.scaling_stop_after > len(self)
        compute_sd = self.sd_stop_after < len(self)
        for iteration in trange(recalculate, desc=self.name, leave=False, position=0):
            if not self.active: continue
            logger.info(' ' * 20 + 'CHAIN {} ITERATION {}'.format(self.name, len(self.accepted)))
            # Save Chain
            self.autosave(50)
            if iteration == recalculate - 1:
                # Acceptance rate
                accept_star = np.mean(self.accepted[-recalculate:])
                self.rates = np.append(self.rates, accept_star)
                # New scaling factor
                if compute_scaling_factor:
                    new_scaling_factor = self.scaling_factor[-1]
                    new_scaling_factor *= np.e ** (accept_star - self.accept_hat)
                    self.scaling_factor = np.append(self.scaling_factor, new_scaling_factor)
                else:
                    new_scaling_factor = 1
                # New COV
                if compute_sd:
                    new_cov_tmp = self.cov.copy()
                    try:
                        sigma_star = np.cov(self.chain[-recalculate:, :].T)
                        new_cov = self.cov.copy() * 0.25 + 0.75 - sigma_star
                        proposed = multinorm(self.values, new_cov * new_scaling_factor ** 2)
                        self.cov = new_cov
                    except Exception as e:
                        print(e)
                        print("Singular COV at", len(self), self.name)
                        print(self.cov)
                        self.cov = new_cov_tmp
                self.sd = new_scaling_factor ** 2 * self.cov

            # Current State
            ll_now = log_likelihood(self.y_now, self.ydata, sigma=self.sigma)

            # try:
            #     proposed = multinorm(self.values, self.sd)
            # except Exception as e:
            #     print(e)
            #     print(self.name, "TURNED OFF", len(self))
            #     self.turn_off()
            #     save_mcmc(self)
            #     continue
            proposed = multinorm(self.values, self.sd)
            guess = proposed.rvs()
            print(guess)
            self.model.update(guess)
            if not self.model.check_proposal():  # Bad guess
                logger.info("bad prior")
                ll_star = -np.inf
                y_star, state_z = self.no_likelihood
                self.model.rollback()
            else:  # Good guess continue MCMC
                try:
                    self.set_stochastics()
                    logger.info('guess: {}'.format(guess))
                    y_star, state_z = self.run_model(7)
                    ll_star = log_likelihood(y_star, self.ydata, self.sigma)
                    logger.info(str(ll_star))
                    if ll_star < -(1e20):
                        logger.warning('bad set for model {} with guess {}'.format(self.name, iteration))
                except Exception as e:
                    logger.info(e)
                    logger.error('exception at model {} PROBABLY S-I-R fail'.format(self.name))
                    y_star, state_z = self.no_likelihood
                    ll_star = -np.inf

            log_r = ll_star - ll_now
            draw = np.random.rand()

            if log_r > np.log(draw):
                self.values = self.model.values
                self.accepted = np.append(self.accepted, 1)
                self.y_now = y_star
            else:
                self.accepted = np.append(self.accepted, 0)

            # Update
            self.chain = np.vstack((self.chain, self.values))
            self.guesses = np.vstack((self.guesses, guess))
            self.mle = np.max((self.mle, ll_now))
            self.yhat_history = np.concatenate((self.yhat_history, y_star[None, :, :]), axis=0)
            self.state_z_history.append(self.state_z)
            self.ll_history = np.vstack((self.ll_history, np.array([ll_now, ll_star])))

    @property
    def no_likelihood(self):
        raise NotImplementedError

    def run_model(self, resolution):
        '''returns the results of the model and the state at last point of time
        :param resolution:
        '''
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError

    def turn_off(self):
        self.active = False

    def autosave(self, every=50, path='./'):
        if len(self) % every == 0:
            save_mcmc(self, path)

    @classmethod
    def load(cls, path):
        mcmc = load_mcmc(path)
        assert isinstance(mcmc, cls), "Loaded file is not of type".format(cls)

    def __len__(self):
        return len(self.chain)

    def __str__(self):
        return (f"{self.name}\n{self.names}\n{self.values}")

    __repr__ = __str__


class Rota(Disease):

    def __init__(self, name, model, populate_values: dict, eq_func):
        # Rota Specific
        extra = {}
        extra['sigma'] = np.array([15662, 31343, 40559, 19608, 6660]).reshape(5, 1)
        extra['state_0'] = collect_state_0(RotaData)
        extra.update(populate_values)
        # Disease
        super(Rota, self).__init__(name, model, extra, eq_func)

    def run_equations(self):
        self.equations()

    @property
    def no_likelihood(self):
        c = -np.inf * np.ones((5, 52 * 9))
        z = COMP._make([-np.inf * np.ones((RotaData.J)).reshape(-1, 1) for _ in COMP._fields])
        return c, z

    def run_model(self, resolution=7):
        prior = self.start - self.years_prior
        c_real = self.equations(steps=resolution * 52, start=prior)
        state_z = COMP._make([comp[:, -1] for comp in c_real])
        c = COMP._make([comp[:, -resolution * 52 * self.end::resolution] for comp in c_real])
        c = make_model_cases(c)
        return c, state_z

    def get_data(self):
        self.ydata = np.genfromtxt('data/cases.csv', delimiter=',', skip_header=1).T
        self.xdata = np.arange(2003, 2012, 1 / 52)
        self.yshape = self.ydata.shape
        self.xshape = self.xdata.shape
        # assert sim1 of x is like y

    def compute_jump(self, scaling_stop_after=10000, sd_stop_after=10000):
        # Model Specific
        self.scaling_factor = np.array([2.4 / np.sqrt(self.d)])
        self.cov = np.diag(np.ones(self.d)) * 1e-7
        self.sd_stop_after = sd_stop_after
        self.scaling_stop_after = scaling_stop_after
        # After updating
        self.sd = self.scaling_factor ** 2 * self.cov


class Chains(object):
    def __init__(self, *args):
        self.chains = list(args)


def make_model_cases(c: COMP):
    union = RotaData.age_union
    long = (c.Im1 + c.Is1) * 7 / RotaData.long_infection_duration
    short = (c.Ia1 + c.Ia2 + c.Im2 + c.Is2 + c.Ia3 + c.Im3 + c.Is3) * 7 / RotaData.short_infection_duration
    I = RotaData.JAPAN_POPULATION * (short + long)
    split = np.vsplit(I, union.cumsum())
    return np.array([xi.sum(axis=0) for xi in split[:-1]])


if __name__ == '__main__':
    logger.info("start")
    # z = Stochastic('z', 0, 1, .5)
    b1 = Stochastic('b1', 0, 2)
    b2 = Stochastic('b2', 0, 2)
    b3 = Stochastic('b3', 0, 2)
    b4 = Stochastic('b4', 0, 2)
    b5 = Stochastic('b5', 0, 2)
    phi = Stochastic('phi', 0, 2 * np.pi)
    vars = [b1, b2, b3, b4, b5, phi]
    extra = {'t': 5, 'scaling_factor': 0.2}
    x = Rota('x', vars, extra, rota_eq)
    x.equations()
    # rota_eq(x)
    print(type(x))
    print(Chains(12, 12, 55, 12).chains)
