from scipy.stats import uniform

from rsv import *


class Disease(object):
    def __init__(self, name, stochastics, **kwargs):
        self.name = name
        self.stochastics = stochastics

        # Constants
        self.accept_hat = 0.23
        self.sigma = 1
        self.state_0 = None
        self.start = None
        self.end = None
        # Initial Values
        self.values = [s.value for s in self.stochastics]
        self.names = [s.name for s in self.stochastics]
        # self.y_hat, self.state_z = self.run_model(self)

        # Chains and Metrics
        self.get_data()
        self.yhat_history = np.zeros(np.append(0, self.yshape))

        self.populate(kwargs)
        self.compute_jump()

    def compute_jump(self):
        # Model Specific

        self.d = len(self.values)
        self.scaling_factor = np.array([2.4 / np.sqrt(self.d)])
        self.cov = np.diag(np.ones(self.d))

        # After updating
        self.sd = self.scaling_factor ** 2 * self.cov

    def update(self):
        '''updates the chains, rates, etc'''

    def populate(self, attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def clinical_data(self, ):
        '''Takes a list of constants and populates them to the model'''
        # Do something with dicts or json...
        # Use populate to populate them

    def sample(self):
        pass

    def run_model(self):
        '''returns the results of the model and the state at last point of time'''
        raise NotImplementedError

    def run_equations(self):
        raise NotImplementedError

    def get_data(self):
        print("GETTING DATA")
        raise NotImplementedError

    def __len__(self):
        return len(self.chain)

    def __str__(self):
        return (f"{self.name}\n{self.names}\n{self.values}")
    __repr__ = __str__


def test_eq():
    print('test eq')


class RSV(Disease):

    def __init__(self, name, stochastics, populate_values, eq_func):
        super(RSV, self).__init__(name, stochastics)
        self.stochastics = stochastics
        self.equations = eq_func
        self.get_data()

        self.populate(populate_values)

    def run_equations(self):
        self.equations()

    def get_data(self):
        self.datax = np.arange(5)
        self.datay = self.datax ** 2
        self.xshape = self.datax.shape
        self.yshape = self.datay.shape
        # assert sim1 of x is like y


class Chains(object):
    def __init__(self, *args):
        self.chains = list(args)


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


if __name__ == '__main__':
    z = Stochastic('z', 0, 1, .5)
    d = {'values'}
    x = RSV('x', [z], {}, test_eq)
    print(type(x))
    print(Chains(12, 12, 55, 12).chains)
