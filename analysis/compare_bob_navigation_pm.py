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

def plot_heads(ax, db, test_entries, heads):
    ax.plot(db.x, db.y)
    x = [db.x[i] for i in test_entries]
    y = [db.y[i] for i in test_entries]
    u = np.cos(heads)
    v = np.sin(heads)

    ax.quiver(x, y, u, v, angles='xy', scale_units='xy') #, scale=10)
    ax.plot(x[0], y[0], 'go')
    ax.axis('equal')

bob_pm = bobnav.PerfectMemory(IM_SIZE)

db = nb.Database(DB_PATH)
train_images = db.read_images(preprocess=ip.resize(*IM_SIZE), to_float=False)
bob_pm.train_route(DB_PATH)

test_entries = range(0, len(db), 50)
test_images = [train_images[i] for i in test_entries]
head_offset = db.heading[test_entries]

t0 = time()
with mp.Pool() as pool:
    bob_heads = np.array(pool.map(lambda im: bob_pm.get_heading(im), test_images))
elapsed = time() - t0
print(f'Took {elapsed} s')

nb_heads = nb.get_ridf_headings(test_images, train_images)
# nb_heads = nb.get_infomax_headings(nb_im, test_images, parallel=False)

_, axes = plt.subplots(1, 2)
plot_heads(axes[0], db, test_entries, nb_heads + head_offset)
plot_heads(axes[1], db, test_entries, bob_heads + head_offset)

# _, ax = plt.subplots()
# ax.plot(nb.normalise180(np.rad2deg(nb_heads)))
# ax.plot(nb.normalise180(np.rad2deg(-bob_heads)))

plt.show()
