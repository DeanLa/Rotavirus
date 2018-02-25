import numpy as np
from scipy.stats import gaussian_kde

from rota import *
import matplotlib.pyplot as plt


def plot_compartment(arr, ages=None, ax=None, **kwargs):
    if ages == None:
        arr = arr.sum(axis=0)
        label = kwargs.get('label', '')
    else:
        d = RotaData()
        if ages == 'all':
            ages = range(d.J)
        arr = arr[ages, :]
        label = [f'{l:.2f}' for l in d.a_l[ages]]
    # PLOT
    if ax:
        fig = ax.figure.canvas
    else:
        fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(arr.T)
    if ages:
        lines = ax.lines
        label = list(label)
        for i, line in enumerate(lines):
            line.set_label(label[i])
            x = line._x[-1] + 0.05
            y = line._y[-1]
            name = line._label
            ax.annotate(name, xy=(x, y), textcoords='data')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return fig, ax


def plot_compartments(obj, compartments=None, ax=None):
    if compartments is None:
        compartments = obj._fields
    if isinstance(compartments, str):
        compartments = compartments.strip().split(' ')

    if ax:
        fig = ax.figure.canvas
    else:
        fig, ax = plt.subplots(figsize=(16, 9))
    for comp in compartments:
        arr = getattr(obj, comp).sum(axis=0)
        label = comp
        ax.plot(arr, label=label)
    for line in ax.lines:
        x = line._x[-1] + 0.05
        y = line._y[-1]
        name = line._label
        ax.annotate(name, xy=(x, y), textcoords='data')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return fig, ax


def plot_against_data(model, data=None, **kwargs):
    if data is None:
        data = model.ydata
    if isinstance(model, Disease):
        model, _ = model.best_run()
    titles = ['Age 0-1', 'Age 1-2', 'Age 2-5', 'Age 5-15', 'Age >15', 'Total']
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))
    axs = np.hstack(axs)
    xaxis = np.arange(data.shape[1])
    for i, ax in enumerate(axs[:-1]):
        ax.plot(xaxis, data[i, :], label='Data', color='red')
        ax.plot(xaxis, model[i, :], label='Model', color='k')
        ax.set_title(titles[i])
        ax.legend(bbox_to_anchor=(0.5, -0.05), ncol=2, mode='expand')

    ax = axs[-1]
    ax.plot(xaxis, data.sum(axis=0), label='Data', color='red')
    ax.plot(xaxis, model.sum(axis=0), label='Model', color='k')
    ax.set_title('Total')
    ax.legend(bbox_to_anchor=(0.5, -0.05), ncol=2, mode='expand')

    return fig, ax


# MCMC
def plot_stoch_vars(mcmc, dist_from=None, chain_from=None):
    # Plot the variables
    # Initialize data frame for later analysis
    assert isinstance(mcmc, Disease), "mcmc should be MCMC object"
    stochastics = mcmc.model.stochastics
    from scipy.stats.mstats import mquantiles
    if not dist_from:
        dist_from = mcmc.tally
    if not chain_from:
        chain_from = mcmc.tally

    # tr_len = mcmc.trace(stochastics[0])[:, None].shape[0]
    # Height is determined by number of variables
    height = 3 * len(stochastics)
    fig, axs = plt.subplots(len(stochastics), 2, figsize=(16, height))

    for i, stoch in enumerate(stochastics):
        try:
            # print str(tr)
            # Prepare Values

            # Plot
            ### Left: Histogram
            tr_val = mcmc.chain[dist_from:, i]  # mcmc.trace(tr)[:, None]
            quants = mquantiles(tr_val, prob=[0.025, 0.25, 0.5, 0.75, 0.975])
            xaxis = np.linspace(tr_val.min(), tr_val.max(), 100)
            ax = axs[i, 0]
            kde = gaussian_kde(tr_val)
            # h = ax.hist(tr_val, histtype='stepfilled', bins=50, label=stoch.name, alpha=0.6)
            yaxis = kde(xaxis)
            ax.plot(xaxis, yaxis, linewidth=4)
            m = max(yaxis) * 1.1
            ax.fill_betweenx([0, m], quants[0], quants[-1], color='lightgreen')
            ax.fill_betweenx([0, m], quants[1], quants[-2], color='darkgreen')
            ax.set_title("{var} = {value:.4f} [{lci:.4f}, {hci:.4f}] ({tmp:.4f})".format(var=stoch.name,
                                                                                         tmp=tr_val.mean(),
                                                                                         value=quants[2],
                                                                                         lci=quants[0],
                                                                                         hci=quants[-1]))
            ax.set_xticks(ax.get_xticks()[::2])  # X 0
            ax.set_yticks(ax.get_yticks()[::2])  # Y 0
            ax.set_ylim([0, m])

            ### Right: Trace
            tr_val = mcmc.chain[chain_from:, i]  # mcmc.trace(tr)[:, None]
            quants = mquantiles(tr_val, prob=[0.025, 0.25, 0.5, 0.75, 0.975])
            ax = axs[i, 1]
            xaxis = np.arange(len(tr_val)) + chain_from
            ax.plot(xaxis, tr_val)
            ax.set_xticks(ax.get_xticks()[::2])  # X 1
            # ax.set_yticks(ax.get_yticks()[::2])  # Y 1
            ax.set_xlim([xaxis[0], xaxis[-1]])
            ax.set_ylim([0, 10])

        except Exception as e:
            print(e)
            print(stoch.name, " excluded")
    # fig.set_tight_layout
    fig.suptitle("Posterior Distribution <|> Convergence")
    return fig, axs


def plot_likelihood_cloud(mcmc: Disease, max_lines=100):
    fig, axs = plt.subplots(1, figsize=(16, 9))
    data = mcmc.ydata.sum(axis=0)
    ax = axs
    w = np.where(mcmc.ll_history[:, 1, ] == mcmc.mle)[0][0]
    best = mcmc.yhat_history[w]
    xaxis = np.arange(len(data)) / 52 + 2003
    ax.scatter(xaxis, data, label='Data', color='red', zorder=5)
    ax.plot(xaxis, best.sum(axis=0), label='MLE Model', color='k', zorder=10)
    ax.set_title('Likelihood Cloud')
    ax.legend(bbox_to_anchor=(0.5, -0.05), ncol=2, mode='expand')

    length = len(mcmc) - mcmc.tally
    every = length // max_lines
    for curr_model in mcmc.yhat_history[mcmc.tally::every]:
        ax.plot(xaxis, curr_model.sum(axis=0), label='Model Realizations', color='grey', zorder=0, alpha=0.04)

    ax.set_ylabel("Cases")  # , fontdict=label_font)
    ax.set_xlabel("Year")  # , fontdict=label_font)
    ax.set_xticks(np.arange(xaxis[0], xaxis[-1], 2))
    ax.set_xlim(xaxis[0], xaxis[-1])
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', which='major')  # , labelsize=20)

    ax.legend()
    return fig, ax


def plot_bars_total(mcmc: Disease, ax=None, best=None):
    if ax:
        fig = ax.figure.canvas
    else:
        fig, ax = plt.subplots(figsize=(16, 9))
    assert isinstance(ax, plt.Axes)
    titles = ['','Age 0-1', 'Age 1-2', 'Age 2-5', 'Age 5-15', 'Age >15']
    if best is None:
        best, _ = mcmc.best_run()
    data = mcmc.ydata
    total_best = best.sum(axis=1)
    total_data = data.sum(axis=1)
    ax.bar(np.arange(len(total_best))-1.2, total_best, width=0.4, label="Model")
    ax.bar(np.arange(len(total_data))-0.8, total_data, width=0.4, label="Data")
    ax.set_xticklabels(titles)
    ax.legend()
    return fig, ax


if __name__ == '__main__':
    import os

    mcmc = Rota.load('../chains/Final.pkl')
    output_dir = os.path.join('..', 'output', '{}')
    # print (mcmc)
    mcmc.tally = 5000
    fig, ax = plot_likelihood_cloud(mcmc, 1500)
    plt.tight_layout()
    fig.savefig(output_dir.format('cloud.png'))
    fig, ax = plot_against_data(mcmc)
    fig.savefig(output_dir.format('against_data.png'))
    fig, ax = plot_stoch_vars(mcmc, 10000,1)
    plt.tight_layout()
    fig.savefig(output_dir.format('chains.png'))
    fig, ax = plot_bars_total(mcmc)
    fig.savefig(output_dir.format('bars.png'))
    plt.show()




    # best, _ = mcmc.best_run()
    # r = best.sum(axis=1) / mcmc.ydata.sum(axis=1)
    # print(r)
    # print(best.sum() / mcmc.ydata.sum())
    # plot_against_data(best, mcmc.ydata)
    # best = best/r.reshape(5,1)
    # print (log_likelihood(best,mcmc.ydata,mcmc.sigma))
    # # fig, ax = plot_bars_total(mcmc)
    # # fig, ax = plot_bars_total(mcmc, best=best)
    # plot_against_data(best, mcmc.ydata)
    # plt.show()