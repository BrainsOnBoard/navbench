import os
import sys
from glob import glob

import numpy as np
import gm_plotting
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(ROOT)

import navbench as nb
import bob_robotics.navigation as bobnav

DBROOT = os.path.join(ROOT, 'datasets/rc_car/rc_car_big')


def get_paths():
    return glob(os.path.join(DBROOT, 'unwrapped_*'))


def load_databases(paths=get_paths(), limits_metres=None):
    return [nb.Database(path, limits_metres=limits_metres, interpolate_xy=True) for path in paths]


@nb.cache_result
def _get_pm_headings(pm, test_df):
    return pm.ridf(test_df)


def to_merc(db):
    mlat, mlon = gm_plotting.utm_to_merc(db.x, db.y, 30, 'U')

    # Convert to x, y
    return mlon, mlat


def get_gps_quality(df):
    return df['GPS quality'].apply(pd.to_numeric, errors='coerce')


class Analysis:
    def __init__(self, train_route: nb.Database, train_skip, preprocess=None):
        self.train_route = train_route
        self.preprocess = preprocess

        self.train_entries = train_route.read_image_entries(
            train_route.iloc[::train_skip], preprocess=preprocess)

        print(f'Training images: {len(self.train_entries)}')

        self.pm = bobnav.PerfectMemory(self.train_entries.image[0].shape)
        self.pm.train(self.train_entries)

    def get_headings(self, test_route, test_skip):
        test_df = test_route.read_image_entries(
            test_route.iloc[:: test_skip],
            preprocess=self.preprocess)
        print(f'Test images: {len(test_df)}')

        test_df = _get_pm_headings(self.pm, test_df)

        nearest = self.train_route.get_nearest_entries(test_df)
        test_df['nearest_train_idx'] = nearest.index
        target_headings = nearest.heading
        dhead = np.array(target_headings) - np.array(test_df.estimated_heading)
        test_df['heading_error'] = np.abs(nb.normalise180(np.rad2deg(dhead)))
        return test_df
