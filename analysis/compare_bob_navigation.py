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
LEARNING_RATE = 0.01
TANH_SCALING_FACTOR = 0.1
SEED = 42

def get_initial_weights(im_size, num_hidden, seed):
    weights, _ = bobnav.InfoMax.generate_initial_weights(
        im_size, seed=seed, num_hidden=num_hidden)
    return weights


def get_ann_bobnav(
        im_size, weights, learning_rate, tanh_scaling_factor, db_path):
    algo = bobnav.InfoMax(
        im_size, weights=weights, learning_rate=learning_rate,
        tanh_scaling_factor=tanh_scaling_factor)
    db = nb.Database(db_path)
    algo.train(db.read_images(preprocess=ip.resize(*IM_SIZE), to_float=False))

    return algo.get_weights()


@nb.cache_result
def get_ann_nb(init_weights, learning_rate, tanh_scaling_factor, train_images):
    algo = nb.InfoMax(learning_rate=learning_rate,
                      tanh_scaling_factor=tanh_scaling_factor, weights=init_weights)
    for im in train_images:
        algo.train(im)

    return algo


def plot_heads(ax, entries, heads):
    ax.plot(entries.x, entries.y)
    u = np.cos(heads)
    v = np.sin(heads)

    ax.quiver(entries.x, entries.y, u, v, angles='xy', scale_units='xy') #, scale=10)
    ax.plot(entries.x[0], entries.y[0], 'go')
    ax.axis('equal')


init_weights = get_initial_weights(
    IM_SIZE, num_hidden=np.prod(IM_SIZE), seed=SEED)

weights = get_ann_bobnav(IM_SIZE, init_weights,
                         LEARNING_RATE, TANH_SCALING_FACTOR, DB_PATH)
bob_im = bobnav.InfoMax(
    IM_SIZE, weights=weights, learning_rate=LEARNING_RATE,
    tanh_scaling_factor=TANH_SCALING_FACTOR)

db = nb.Database(DB_PATH)
train_images = db.read_images(preprocess=ip.resize(*IM_SIZE), to_float=False)
nb_im = get_ann_nb(init_weights, LEARNING_RATE, TANH_SCALING_FACTOR, train_images)

test_entries = db.iloc[0::50]
test_images = [train_images[i] for i in test_entries.index]

t0 = time()
data = bob_im.get_ridf_data(test_images)
bob_heads = data.heading.to_numpy()
elapsed = time() - t0
print(f'Took {elapsed} s')

nb_heads = nb.get_infomax_headings(nb_im, test_images, parallel=False)

print(data)

_, axes = plt.subplots(1, 2)
plot_heads(axes[0], test_entries, nb_heads + test_entries.heading)
plot_heads(axes[1], test_entries, bob_heads + test_entries.heading)

plt.show()
