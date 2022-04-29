# %%
import os
ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
import sys
sys.path.append(ROOT)

import rc_car_big
import matplotlib.pyplot as plt
from navbench import imgproc as ip
import numpy as np

TRAIN_SKIP = 1
TEST_SKIP = 80
# PREPROC = None
PREPROC = ip.resize(45, 180)
BIN_WIDTH = 100  # metres

paths = rc_car_big.get_paths()
dbs = rc_car_big.load_databases(paths) #[0:8])

train_route = dbs[0]
test_routes = dbs[1:]

analysis = rc_car_big.Analysis(train_route, train_skip=TRAIN_SKIP, preprocess=PREPROC)
for test_route in test_routes:
    test_df = analysis.get_headings(test_route, TEST_SKIP)
    dist = test_df.distance.to_numpy()

    num_bins = int(dist[-1] // BIN_WIDTH)
    if dist[-1] % BIN_WIDTH: num_bins += 1

    test_df['bin'] = None
    data = []
    for i in range(num_bins):
        sel = np.logical_and(dist >= (BIN_WIDTH * i), dist < (BIN_WIDTH * (i + 1)))
        test_df['bin'] = np.where(sel, BIN_WIDTH * (i + 1), test_df['bin'])
        data.append(np.minimum(90, test_df[sel].heading_error))

    bindat = test_df.bin
    bins = test_df.bin.unique()
    labels = bins  # [f'{bin_rng[i]:.0f}-{bin_rng[i+1]:.0f}' for i in range(NUM_BINS)]

    _, ax = plt.subplots()
    ax.boxplot(data, labels=labels)
    ax.set_title(test_route.name)
    ax.set_ylim(bottom=0)

plt.show()
