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
