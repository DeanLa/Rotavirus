import matplotlib.pyplot as plt
from rota import *

# mcmc = Rota.load('./chains/mcmc180208.pkll')
mcmc = Rota.load('./mcmc.pkl')
best, _ = mcmc.best_run()
fig, ax = plot_against_data(best,mcmc.ydata)
print (mcmc.accepted.mean())
print (len(mcmc))
# plt.tight_layout()
plt.show()



x=1