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

# fig, ax = plt.subplots(6)
#
# for i in range(16):
#     m=Rota.load('./chains/c0209/mcmc_mp_{}.pkl'.format(i))
#     print ('='*80)
#     print(m.name)
#     print ("Length {}".format(len(m)))
#     print("MLE: {}".format(m.mle))
#     if m.mle > -150: print('*'*50)
#     best, _ = m.best_run()
#     for i in range(6):
#         ax[i].plot(m.chain[0:,i])
# plt.show()

# for i in range(16):
#     m=Rota.load('./chains/b0209/mcmc_mp_{}.pkl'.format(i))
m=Rota.load('./chains/c0209/mcmc_mp_{}.pkl'.format(14))
best, _ = m.best_run()
print (best.sum(axis=1) / m.ydata.sum(axis=1))
print (best.sum()/ m.ydata.sum())

plot_against_data(best, m.ydata)
plt.show()
# plot_likelihood_cloud(mcmc,2000); plt.show()
x=1