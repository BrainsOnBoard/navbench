import os
import sys
from glob import glob
from time import perf_counter

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(ROOT)

import navbench as nb
import bob_robotics.navigation as bobnav

DBROOT = os.path.join(ROOT, 'datasets/rc_car/rc_car_big')


def get_paths():
    return glob(os.path.join(DBROOT, 'unwrapped_*'))


def load_databases(paths=get_paths(), limits_metres=None):
    return [nb.Database(path, limits_metres=limits_metres, interpolate_xy=True) for path in paths]

class Analysis:
    def __init__(
            self, train_route: nb.Database, train_skip, preprocess=None,
            to_float=False):
        self.train_route = train_route
        self.to_float = to_float
        self.preprocess = preprocess

        self.train_entries = train_route.read_image_entries(
            train_route.iloc[::train_skip], to_float=to_float, preprocess=preprocess)

        print(f'Training images: {len(self.train_entries)}')

        self.pm = bobnav.PerfectMemory(self.train_entries.image[0].shape[::-1])
        self.pm.train(self.train_entries)

    def get_headings(self, test_route, test_skip):
        test_df = test_route.read_image_entries(
            test_route.iloc[:: test_skip],
            to_float=self.to_float, preprocess=self.preprocess)
        print(f'Test images: {len(test_df)}')

        test_df = test_df.join(self.pm.ridf(test_df))

        nearest = self.train_route.get_nearest_entries(test_df)
        test_df['nearest_train_idx'] = nearest.index
        target_headings = nearest.heading
        dhead = np.array(target_headings) - np.array(test_df.estimated_heading)
        test_df['heading_error'] = np.abs(nb.normalise180(np.rad2deg(dhead)))
        return test_df
