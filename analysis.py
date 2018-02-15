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
folder = '0216Final'
fig, ax = plt.subplots(7)
mymax = -np.inf
best_model = -1
for i in range(12):
    m = Rota.load('./chains/{}/mcmc_mp_{}.pkl'.format(folder,i))
    print('=' * 80)
    print(m.name)
    print("Length {}".format(len(m)))
    mle = m.mle
    print("MLE: {}".format(mle))
    if mle > mymax:
        print ("WHAT")
        mymax = mle
        best_model = i
    if mle > -150: print('*' * 50)
    best, _ = m.best_run()
    for j in range(6):
        ax[j].plot(m.chain[0:, j])
# plt.show()
print('=' * 80)
print('=' * 80)

# for i in range(16):
#     m=Rota.load('./chains/0209b/mcmc_mp_{}.pkl'.format(i))
print('BEST {}'.format(best_model))
m = Rota.load('./chains/{}/mcmc_mp_{}.pkl'.format(folder, best_model))
best, _ = m.best_run()
print(best.sum(axis=1) / m.ydata.sum(axis=1))
print(best.sum() / m.ydata.sum())

plot_against_data(best, m.ydata)
plot_stoch_vars(m)
plt.show()
# plot_likelihood_cloud(mcmc,2000); plt.show()
x = 1
