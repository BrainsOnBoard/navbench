#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__) + "/..")

import matplotlib.pyplot as plt
import numpy as np
import navbench.imgproc as ip
import navbench as nb
import bob_robotics.navigation as bobnav
import pathos.multiprocessing as mp
from time import time

IM_SIZE = (90, 10)
DB_ROOT = "datasets/ant_world/routes"
DB_PATH = DB_ROOT + "/ant1_route1"
LEARNING_RATE = 0.01
TANH_SCALING_FACTOR = 0.1
SEED = None

def get_initial_weights(im_size, num_hidden, seed):
    weights, _ = bobnav.InfoMax.generate_initial_weights(
        im_size, seed=seed, num_hidden=num_hidden)
    return weights


@nb.cache_result
def get_ann_bobnav(
        im_size, weights, learning_rate, tanh_scaling_factor, db_path):
    algo = bobnav.InfoMax(
        im_size, weights=weights, learning_rate=learning_rate,
        tanh_scaling_factor=tanh_scaling_factor)
    algo.train_route(db_path)
    return algo.get_weights()


@nb.cache_result
def get_ann_nb(weights, learning_rate, tanh_scaling_factor, train_images):
    algo = nb.InfoMax(learning_rate=learning_rate,
                      tanh_scaling_factor=tanh_scaling_factor, weights=weights)
    for im in train_images:
        algo.train(im)

    return algo


def plot_heads(ax, db, test_entries, heads):
    ax.plot(db.x, db.y)
    x = [db.x[i] for i in test_entries]
    y = [db.y[i] for i in test_entries]
    u = np.cos(heads)
    v = np.sin(heads)

    ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=2)
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
nb_im = get_ann_nb(weights, LEARNING_RATE, TANH_SCALING_FACTOR, train_images)

test_entries = range(0, len(db), 50)
test_images = [train_images[i] for i in test_entries]

t0 = time()
with mp.Pool() as pool:
    bob_heads = pool.map(lambda im: bob_im.get_heading(im), test_images)
elapsed = time() - t0
print(f'Took {elapsed} s')

nb_heads = nb.get_infomax_headings(nb_im, test_images)

_, axes = plt.subplots(1, 2)
plot_heads(axes[0], db, test_entries, nb_heads)
plot_heads(axes[1], db, test_entries, bob_heads)

plt.show()
