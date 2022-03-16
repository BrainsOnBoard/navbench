import matplotlib.pyplot as plt
import numpy as np


def anglequiver(ax, x, y, theta, **kwargs):
    u = np.cos(theta)
    v = np.sin(theta)
    return ax.quiver(x, y, u, v, angles='xy', **kwargs) #, scale_units='xy')
