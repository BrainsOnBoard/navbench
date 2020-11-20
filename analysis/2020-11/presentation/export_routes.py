#!/usr/bin/python3
import sys
sys.path.append('../..')

import os
import matplotlib.pyplot as plt
import pandas as pd
import navbench as nb

dbroot = '../../../datasets/rc_car/Stanmer_park_dataset'

# Plot UTM coordinates
def plot_route(ax, dpath):
    db = nb.Database(os.path.join(dbroot, dpath))
    ax.plot(db.entries['x'], db.entries['y'], ':')

def get_dataset_path(date, i):
    return '%s/dataset%d' % (date, i)

def get_dataset_paths(date):
    i = 1
    paths = []
    while True:
        path = get_dataset_path(date, i)
        if not os.path.exists(os.path.join(dbroot, path)):
            break
        paths.append(path)
        i += 1
    return paths

def plot_routes(paths):
    _, ax = plt.subplots()

    for path in paths:
        print('Loading ' + path)
        plot_route(ax, path)

    # ax.legend(paths)
    ax.axis('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

plot_routes(get_dataset_paths('0411'))

plt.savefig('routes.svg')
