import os
import sys
from glob import glob

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
sys.path.append(ROOT)

import navbench as nb

DBROOT = os.path.join(ROOT, 'datasets/rc_car/rc_car_big')


def get_paths():
    return glob(os.path.join(DBROOT, 'unwrapped_*'))


def load_databases(paths=get_paths(), limits_metres=None):
    return [nb.Database(path, limits_metres=limits_metres, interpolate_xy=True) for path in paths]


def get_database_subset(db, skip):
    assert not db.x.hasnans
    return db.iloc[::skip].copy()

class Analysis:
    def __init__(
            self, train_route: nb.Database, train_skip, preprocess=None,
            to_float=False):
        self.train_route = train_route
        self.to_float = to_float
        self.preprocess = preprocess

        self.train_entries = get_database_subset(train_route, train_skip)
        self.train_entries['image'] = train_route.read_images(
            self.train_entries, to_float=to_float, preprocess=preprocess)

        print(f'Training images: {len(self.train_entries)}')

    def get_headings(self, test_route, test_skip):
        test_df = get_database_subset(test_route, test_skip)
        test_df['image'] = test_route.read_images(
            test_df, to_float=self.to_float, preprocess=self.preprocess)
        print(f'Test images: {len(test_df)}')

        headings_df = nb.get_ridf_headings_and_snap(test_df, self.train_entries)
        test_df = test_df.join(headings_df)

        nearest = self.train_route.get_nearest_entries(test_df)
        test_df['nearest_train_idx'] = nearest.index
        target_headings = nearest.heading
        dhead = np.array(target_headings) - np.array(test_df.estimated_heading)
        test_df['heading_error'] = np.abs(nb.normalise180(np.rad2deg(dhead)))
        return test_df
