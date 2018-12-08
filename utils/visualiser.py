from matplotlib import pyplot as plt
import numpy as np


class PlotManager:
    def __init__(self):
        pass

    def update(self, x, y):
        print(x, y)
        plt.scatter(x, y)
        plt.show()
        plt.pause(1e-7)


if __name__ == '__main__':
    pm = PlotManager()
    for i in range(-20, 20):
        pm.update(i, i * i)
