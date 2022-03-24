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
import numpy as np

TRAIN_SKIP = 40
TEST_SKIP = 40
# PREPROC = None
PREPROC = ip.resize(180, 45)

paths = rc_car_big.get_paths()
dbs = rc_car_big.load_databases(paths[0:4])  #, limits_metres=(0, 200))

train_route = dbs[0]
test_routes = dbs[1:]

def to_merc(db):
    mlat, mlon = gm_plotting.utm_to_merc(db.x, db.y, 30, 'U')

    # Convert to x, y
    return mlon, mlat

train_x, train_y = to_merc(train_route)
analysis = rc_car_big.Analysis(train_route, train_x, train_y, train_skip=TRAIN_SKIP, preprocess=PREPROC)
_, ax0 = plt.subplots()
ax0.plot(train_x, train_y, label='Training route')
_, ax1 = plt.subplots()

for test_route in test_routes:
    test_entries, headings, nearest_train_entries, heading_error = analysis.get_headings(
        test_route, TEST_SKIP)

    label = test_route.name.replace('unwrapped_', '')
    x, y = to_merc(test_route)
    lines = ax0.plot(x, y, '--', label=label)
    colour = lines[0].get_color()
    ax0.plot(x[0], y[0], 'o', color=colour)

    nb.anglequiver(
        ax0, x[test_entries], y[test_entries],
        # scale=1e6, scale_units='xy',
        headings, color=colour, zorder=lines[0].zorder + 1,
        alpha=0.8)

    # Cap at 90°
    heading_error = np.minimum(heading_error, 90)
    ax1.scatter(train_route.distance[nearest_train_entries], heading_error,
                label=label, alpha=0.5, marker='.')
    ax1.set_xlabel("Distance along training route (m)")
    ax1.set_ylabel("Heading error (°)")

gm_plotting.APIClient().add_satellite_image_background(ax0)

ax0.legend()

ax1.legend()
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)

plt.show()
