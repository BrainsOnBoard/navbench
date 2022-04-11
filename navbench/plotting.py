import matplotlib.pyplot as plt
import numpy as np


def anglequiver(ax, x, y, theta, invert_y=False, **kwargs):
    u = np.cos(theta)
    v = np.sin(theta)
    if invert_y:
        v = -v
    return ax.quiver(x, y, u, v, angles='xy', **kwargs) #, scale_units='xy')
