import numpy as np
import pickle
from rota import *

# Math
def mse(x, y):
    res = (x - y) ** 2
    return res.mean()

def log_likelihood(model, data, sigma=1, noise=150):
    diff = (model - data) ** 2
    diff[data > noise] = 0
    LL = -diff / (2 * sigma ** 2)
    return LL.sum()



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
    name = obj['name']
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
    return obj['name']


def load_mcmc(path='./mcmc.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

