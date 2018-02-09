import matplotlib.pyplot as plt
from rota import *

# mcmc = Rota.load('./chains/mcmc180208.pkll')
# mcmc = Rota.load('./chains/180209mcmc.pkl')
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
    m=Rota.load('./chains/b0209/mcmc_mp_{}.pkl'.format(i))
    print ('='*80)
    print(m.name)
    print ("Length {}".format(len(m)))
    print("MLE: {}".format(m.mle))
    if m.mle > -150: print('*'*50)

m=Rota.load('./chains/b0209/mcmc_mp_14.pkl')
best, _ = m.best_run()
plot_against_data(best, m.ydata)
# plot_likelihood_cloud(mcmc,2000); plt.show()
x=1