import os
ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
import sys
sys.path.append(ROOT)

import navbench as nb
import numpy as np
from glob import glob
DBROOT = os.path.join(ROOT, 'datasets/rc_car/rc_car_big')


def get_paths():
    return glob(os.path.join(DBROOT, 'unwrapped_*'))


def load_databases(paths=get_paths(), limits_metres=None):
    return [nb.Database(path, limits_metres=limits_metres, interpolate_xy=True, csvFileName='database_entries_processed.csv') for path in paths]


def get_valid_entries(db, skip):
    lst = [i for i, x in enumerate(db.x) if not np.isnan(x)]
    return lst[::skip]

class Analysis:
    def __init__(self, train_route, train_skip, to_float=False):
        self.train_route = train_route
        self.train_entries = get_valid_entries(train_route, train_skip)
        self.train_images = train_route.read_images(self.train_entries, to_float=to_float)
        self.to_float = to_float

        # TODO: Could do e.g. medianfilt over these headings
        self.train_headings_all = np.arctan2(np.diff(train_route.y), np.diff(train_route.x))
        self.train_headings_all = np.append(self.train_headings_all, [self.train_headings_all[-1]])
        self.train_headings = self.train_headings_all[self.train_entries]

        print(f'Training images: {len(self.train_images)}')

    def get_headings(self, test_route, test_skip):
        test_entries = get_valid_entries(test_route, test_skip)
        test_images = test_route.read_images(test_entries, to_float=self.to_float)
        print(f'Test images: {len(test_images)}')

        headings, best_snaps = nb.get_ridf_headings_and_snap(test_images, self.train_images)
        headings += self.train_headings[best_snaps]

        nearest_train_entries = self.train_route.get_nearest_entries(test_route.x[test_entries], test_route.y[test_entries])
        target_headings = self.train_headings_all[nearest_train_entries]
        heading_error = np.abs(nb.normalise180(np.rad2deg(target_headings - headings)))
        return test_entries, headings, nearest_train_entries, heading_error
