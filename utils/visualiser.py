from matplotlib import pyplot as plt
import numpy as np


def moving_average(data, window_size):
    return np.convolve(data, np.ones((window_size,)) / float(window_size), mode='valid')


class DataPlotter:
    class Plot:
        def __init__(self, name, xlim, ylim, ylabel):
            self.name = name
            self.xlim = xlim
            self.ylim = ylim
            self.ylabel = ylabel

            self.f, self.p = plt.subplots(1)

    def __init__(self):
        self.plots = {}

    def add_plot(self, name, xlim, ylim, ylabel):
        f, p = plt.subplots(1)
        p.set_xlim(xlim)
        p.set_ylim(ylim)
        p.set_xlabel('epochs')
        p.set_ylabel(ylabel)
        self.plots[name] = (f, p)

    # def update_lot(self, name, values):
    #     f, p = self.plots[name]
    #
    #     p.clear()
    #     p.plot(values)
    #     p.set_xlabel('epochs')
    #     p.set_ylabel('loss')

    def update_plot(self, name, i, value):
        f, p = self.plots[name]
        p.scatter(i, value)
        plt.show()
        plt.pause(1e-7)


class PlotManager:
    def __init__(self, xmin, xmax, ymin, ymax):
        axes = plt.gca()
        axes.set_xlim([xmin, xmax])
        axes.set_ylim([ymin, ymax])

    def update(self, x, y):
        print(x, y)
        plt.scatter(x, y)
        plt.show()
        plt.pause(1e-7)

    def plot_values(self, x, y):
        plt.scatter(x, y)
        plt.show()
        plt.pause(1e-7)


if __name__ == '__main__':
    import random
    import time

    pm = DataPlotter()
    pm.add_plot('test', (0, 10), (0, 2), 'test')
    res = []
    for i in range(10):
        res.append(random.random())
        pm.update_plot('test', i, res[-1])
