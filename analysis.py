import matplotlib.pyplot as plt
from rota import *

# mcmc = Rota.load('./chains/mcmc180208.pkll')
# mcmc = Rota.load('./mcmc.pkl')
# mcmc.tally=5000
# best, _ = mcmc.best_run()
# fig, ax = plot_against_data(best,mcmc.ydata)
# plot_stoch_vars(mcmc)
# print (mcmc.accepted.mean())
# print (len(mcmc))
# plt.tight_layout()
# plt.show()


for i in range(16):
    m=Rota.load('./chains/a0209/mcmc_mp_mcmc_{}.pkl'.format(i))
    print(m.name)
    print(m.mle)
    if m.mle > -150: print('*'*50)

x=1