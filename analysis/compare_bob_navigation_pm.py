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
train_entries = db.read_image_entries(preprocess=ip.resize(*IM_SIZE), to_float=False)
train_images = train_entries.image.to_list()
bob_pm.train(train_entries)

test_entries = db.read_image_entries(db.iloc[::50], preprocess=ip.resize(*IM_SIZE), to_float=False)
test_images = test_entries.image.to_list()

t0 = time()
data = bob_pm.ridf(test_entries)
bob_heads = data.estimated_heading.to_numpy()
elapsed = time() - t0
print(f'Took {elapsed} s')

nb_heads = nb.get_ridf_headings(test_images, train_images)
nb_heads += test_entries.heading

_, axes = plt.subplots(1, 2)
plot_heads(axes[0], test_entries, nb_heads)
bh2 = bob_heads + test_entries.heading
plot_heads(axes[1], test_entries, bob_heads)

plt.show()
