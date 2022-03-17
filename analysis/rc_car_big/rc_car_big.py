import os
ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
import sys
sys.path.append(ROOT)

import navbench as nb
from glob import glob
DBROOT = os.path.join(ROOT, 'datasets/rc_car/rc_car_big')


def get_paths():
    return glob(os.path.join(DBROOT, 'unwrapped_*'))


def load_databases(paths=get_paths(), limits_metres=None):
    return [nb.Database(path, limits_metres=limits_metres, interpolate_xy=True, csvFileName='database_entries_processed.csv') for path in paths]
