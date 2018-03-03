import numpy as np
# import pickle
import dill as pickle
from rota import *


# Math
def mse(x, y):
    res = (x - y) ** 2
    return res.mean()


def normal_log_likelihood(model, data, sigma=57460, noise=np.inf):
    sigma = sigma.astype('int64')
    # sigma = np.sqrt((sigma ** 2) /52)
    diff = (model - data) ** 2
    # diff[data > noise] = 0
    LL = - diff / (2 * sigma ** 2)
    return LL.sum()


def normal_pct_log_likelihood(model, data, sigma=57460, noise=np.inf):
    sigma = sigma.astype('int64')
    r = sigma.T / data.max(axis=1)
    sigma = r.T * data
    # sigma = np.sqrt((sigma ** 2) /52)
    diff = (model - data) ** 2
    # diff[data > noise] = 0
    LL = - diff / (2 * sigma ** 2)
    return LL.sum()


def beta_log_likelihood(model, data, alpha=42.34, beta=260):
    '''https://en.wikipedia.org/wiki/Beta_distribution#Maximum_likelihood'''
    X = model / data
    X[:2, :] *= 1772.38 / 2195.78
    X[X > 1] = 1
    LL = (alpha - 1) * np.log(X) + (beta - 1) * np.log(1 - X)
    return LL.sum()


log_likelihood = beta_log_likelihood


def nums(scalar, amount):
    return np.ones(amount) * scalar


def concat_nums(*args):
    n = len(args)
    assert n % 2 == 0, "Even number of args NOT GOOD: {}".format(n)
    iter = np.arange(0, n, 2)
    pairs = []
    for i in iter:
        pairs.append(nums(args[i], args[i + 1]))
    return np.concatenate(pairs)


def gelman_rubin(chains):
    """http://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/"""
    M = len(chains)
    assert M >= 2, "test for less than 2 is redundunt"
    params = chains[0].shape[1]  # Asumming chains have equal params

    N = len(chains[0][:, 0])  # Asumming chains have equal length
    B = np.zeros(params)  # Init for values of params
    W = np.zeros(params)  # Init for values of parmas
    for i in range(params):
        means = [np.mean(chain[:, i]) for chain in chains]
        mean_of_means = np.mean(means)
        variances = [np.var(chain[:, i]) for chain in chains]
        B[i] = N * np.var(means)
        # B = (N / M - 1) *
        W[i] = np.mean(variances)
    V = (1 - (1 / N)) * W + (1 / N) * B
    R = np.sqrt(V / W)
    return R


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


# I/O Operations
def save_mcmc(obj, path='./'):
    # return
    name = obj.name
    save_path = path + name + '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    try:
        save_path = './backup/' + path.replace('/', '-') + name + '.pkl'
        with open(save_path, 'wb') as f2:
            pickle.dump(obj, f2, pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError as e:
        print(e)
        print("NO BACKUP SAVE")
    return obj.name


def load_mcmc(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
