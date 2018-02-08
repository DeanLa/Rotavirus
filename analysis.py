import matplotlib.pyplot as plt
from rota import *

# mcmc = Rota.load('./chains/mcmc180208.pkll')
mcmc = Rota.load('./mcmc.pkl')
mcmc.tally=5000
best, _ = mcmc.best_run()
fig, ax = plot_against_data(best,mcmc.ydata)
plot_stoch_vars(mcmc)
print (mcmc.accepted.mean())
print (len(mcmc))
plt.tight_layout()
plt.show()



x=1