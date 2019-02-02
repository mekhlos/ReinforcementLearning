import numpy as np


def moving_average(data, window_size):
    return np.convolve(data, np.ones((window_size,)) / float(window_size), mode='valid')
