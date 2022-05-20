# %%
import os
ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
import sys
sys.path.append(ROOT)

import gm_plotting
import rc_car_big
import matplotlib.pyplot as plt
import navbench as nb
from navbench import imgproc as ip
from scipy.signal import medfilt
import numpy as np

TRAIN_SKIP = 10
TEST_SKIP = 80
IM_SIZE = (45, 180)
PREPROC = 'None'

paths = rc_car_big.get_paths()

def distance2d(entry1, entry2):
    diff = np.array([entry1.x, entry1.y]) - np.array([entry2.x, entry2.y])
    if diff.ndim == 1:
        return np.hypot(*diff)
    else:
        return np.hypot(diff[0, :], diff[1, :])

def plot_train_route(analysis, train_skip, preprocess):
    train_x, train_y = rc_car_big.to_merc(analysis.train_route)
    ax0.plot(train_x, train_y, label='Training route')

def do_plots(train_route, test_route, df, preprocess, params):
    label = test_route.name.replace('unwrapped_', '')
    x, y = rc_car_big.to_merc(test_route)
    lines = ax0.plot(x, y, '--', label=label)
    colour = lines[0].get_color()
    ax0.plot(x[0], y[0], 'o', color=colour)

    nb.anglequiver(
        ax0, x[df.index], y[df.index],
        # scale=1e6, scale_units='xy',
        df.heading, color=colour, zorder=lines[0].zorder + 1,
        alpha=0.8, invert_y=True)

    # Cap at 90°
    err = np.minimum(df.heading_error, 90)
    dist_along_train = train_route.distance[df.nearest_train_idx]

    ax1.scatter(dist_along_train, err, label=label, alpha=0.5, marker='.')
    ax1.set_xlabel("Distance along training route (m)")
    ax1.set_ylabel("Heading error (°)")

    ax2.plot(dist_along_train, medfilt(err, kernel_size=5), label=label)
    ax2.set_xlabel("Distance along training route (m)")
    ax2.set_ylabel("Heading error (°)")

    dist_to_train = distance2d(df, train_route.loc[df.nearest_train_idx])
    ax3.scatter(dist_to_train, err, label=label, alpha=0.5, marker='.')
    ax3.set_xlabel("Distance to training route (m)")
    ax3.set_ylabel("Heading error (°)")

def finalise_plots(params):
    gm_plotting.APIClient().add_satellite_image_background(ax0)

    ax0.legend(bbox_to_anchor=(1, 1))

    ax1.legend(bbox_to_anchor=(1, 1))
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax2.legend(bbox_to_anchor=(1, 1))
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    ax3.legend(bbox_to_anchor=(1, 1))
    ax3.set_xlim(0, 20)
    ax3.set_ylim(bottom=0)

    plt.show()


_, ax0 = plt.subplots()
_, ax1 = plt.subplots()
_, ax2 = plt.subplots()
_, ax3 = plt.subplots()

rc_car_big.run_analysis(paths[0:1], paths[1:], [TRAIN_SKIP], [TEST_SKIP], [IM_SIZE], [PREPROC], train_hook=plot_train_route, test_hook=do_plots, post_test_hook=finalise_plots)
