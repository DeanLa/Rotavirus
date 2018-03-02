from rota import *
import pandas as pd
import numpy as np

m = Rota.load('./chains/Final.pkl')
m.eq_func = rota_eq


sims = 5
reduct = np.random.rand(sims, 3)
rho_list = [reduct]
for i in range(3):
    tmp = np.random.rand(sims, 3)
    tmp = tmp / tmp.sum(axis=1).reshape(-1,1)
    rho_list.append(tmp)

result = np.hstack(rho_list)
print(result)
s = Simulation(m, result)