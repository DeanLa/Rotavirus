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


def plot_against_data(model, data):
    titles = ['Age 0-1', 'Age 1-2', 'Age 2-5', 'Age 5-5', 'Age >15', 'Total']
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
def plot_stoch_vars(mcmc, which=None):
    # Plot the variables
    # Initialize data frame for later analysis
    assert isinstance(mcmc, Disease), "mcmc should be MCMC object"
    # if which:
    #     stoch = sorted(which)
    # else:
    #     stoch = sorted([str(v) for v in mcmc.stochastics])
    stochastics = mcmc.model.stochastics
    from scipy.stats.mstats import mquantiles

    # tr_len = mcmc.trace(stochastics[0])[:, None].shape[0]
    # Height is determined by number of variables
    height = 3 * len(stochastics)
    fig, axs = plt.subplots(len(stochastics), 2, figsize=(16, height))

    for i, stoch in enumerate(stochastics):
        try:
            # print str(tr)
            # Prepare Values
            tr_val = mcmc.chain[mcmc.tally:, i]  # mcmc.trace(tr)[:, None]
            quants = mquantiles(tr_val, prob=[0.025, 0.25, 0.5, 0.75, 0.975])

            # Plot
            ### Left: Histogram
            ax = axs[i, 0]
            h = ax.hist(tr_val, histtype='stepfilled', bins=50, label=stoch.name, alpha=0.6)
            m = max(h[0])
            ax.fill_betweenx([0, m + 5], quants[0], quants[-1], color='lightgreen')
            ax.fill_betweenx([0, m + 5], quants[1], quants[-2], color='darkgreen')
            ax.set_title("{var} = {value:.4f} [{lci:.4f}, {hci:.4f}] ({tmp:.4f})".format(var=stoch.name,
                                                                                                tmp=tr_val.mean(),
                                                                                                value=quants[2],
                                                                                                lci=quants[0],
                                                                                                hci=quants[-1]))
            ax.set_xticks(ax.get_xticks()[::2])  # X 0
            ax.set_yticks(ax.get_yticks()[::2])  # Y 0
            ax.set_ylim([0, m + 5])
            # print axs[i,1].get_yticks()[::2]
            ### Right: Trace
            ax = axs[i, 1]
            xaxis = np.arange(len(tr_val)) + mcmc.tally
            ax.plot(xaxis, tr_val)
            ax.set_xticks(ax.get_xticks()[::2])  # X 1
            ax.set_yticks(ax.get_yticks()[::2])  # Y 1
            ax.set_xlim([xaxis[0], xaxis[-1]])

        except Exception as e:
            print (e)
            print(stoch.name, " excluded")
    # fig.set_tight_layout
    fig.suptitle("Posterior Distribution <|> Convergence")
    return fig, axs
