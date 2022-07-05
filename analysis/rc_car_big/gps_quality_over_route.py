# %%
import itertools

import gm_plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rc_car_big
from matplotlib.collections import LineCollection


def running_mean(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode="valid")


paths = rc_car_big.get_paths()
dbs = rc_car_big.load_databases(paths)

map_client = gm_plotting.APIClient()
fig, axes = plt.subplots(4, 4)
for db, ax in zip(dbs, itertools.chain(*axes)):
    ax.set_title(db.name.replace("unwrapped_", ""))
    ax.set_xticks(())
    ax.set_yticks(())

    x, y = rc_car_big.to_merc(db)
    gps = db.entries["GPS quality"].apply(pd.to_numeric, errors="coerce")
    valids = ~np.isnan(gps)
    x = x[valids]
    y = y[valids]
    gps = gps[valids]

    # Create a set of line segments so that we can color them individually This
    # creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line
    # collection needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, 5)
    lc = LineCollection(segments, cmap="hot", norm=norm)

    # Set the values used for colormapping. Take a running mean of GPS quality
    lc.set_array(running_mean(gps, 5))
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)

    # Set the axis limits to something sensible
    ax.axis("equal")

    map_client.add_satellite_image_background(ax)

plt.show()
