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
    titles = ['Age 0-1','Age 1-2','Age 2-5','Age 5-5','Age >15','Total']
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))
    axs = np.hstack(axs)
    xaxis = np.arange(data.shape[1])
    for i, ax in enumerate(axs[:-1]):
        ax.plot(xaxis, data[i, :], label='Data',color='red')
        ax.plot(xaxis, model[i, :], label='Model',color='k')
        ax.set_title(titles[i])
        ax.legend(bbox_to_anchor=(0.5, -0.05), ncol=2,mode='expand')

    ax=axs[-1]
    ax.plot(xaxis, data.sum(axis=0), label='Data',color='red')
    ax.plot(xaxis, model.sum(axis=0),label='Model',color='k')
    ax.set_title('Total')
    ax.legend(bbox_to_anchor=(0.5,-0.05), ncol=2, mode='expand')

    return fig, ax
