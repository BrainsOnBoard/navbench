#!/usr/bin/env python3
import sys
import os
BOB_PATH = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(BOB_PATH)

import matplotlib.pyplot as plt
import numpy as np
import navbench.imgproc as ip
import navbench as nb
import bob_robotics.navigation as bobnav
import pathos.multiprocessing as mp
from time import time

IM_SIZE = (90, 10)
DB_ROOT = os.path.join(BOB_PATH, "datasets/rc_car")
DB_PATH = os.path.join(DB_ROOT, "2020-11-04/unwrapped_dataset1")

def plot_heads(ax, entries, heads):
    ax.plot(entries.x, entries.y)
    u = np.cos(heads)
    v = np.sin(heads)

    ax.quiver(entries.x, entries.y, u, v, angles='xy', scale_units='xy') #, scale=10)
    ax.plot(entries.x[0], entries.y[0], 'go')
    ax.axis('equal')

bob_pm = bobnav.PerfectMemory(IM_SIZE)

db = nb.Database(DB_PATH)
train_images = db.read_images(preprocess=ip.resize(*IM_SIZE), to_float=False)
bob_pm.train(train_images)

test_entries = db.iloc[0::50]
test_images = [train_images[i] for i in test_entries.index]

t0 = time()
data = bob_pm.get_ridf_data(test_images)
bob_heads = data.heading.to_numpy()
elapsed = time() - t0
print(f'Took {elapsed} s')

nb_heads = nb.get_ridf_headings(test_images, train_images)

_, axes = plt.subplots(1, 2)
plot_heads(axes[0], test_entries, nb_heads + test_entries.heading)
bh2 = bob_heads + test_entries.heading
plot_heads(axes[1], test_entries, bob_heads + test_entries.heading)

plt.show()
