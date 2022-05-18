import os
from glob import glob

import bob_robotics.navigation as bobnav
from bob_robotics.navigation import imgproc as ip
import numpy as np
import pandas as pd

import gm_plotting

ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
DBROOT = os.path.join(ROOT, 'datasets/rc_car/rc_car_big')


def get_paths():
    return glob(os.path.join(DBROOT, 'unwrapped_*'))


def load_databases(paths=get_paths(), limits_metres=None):
    dbs = [bobnav.Database(path, limits_metres=limits_metres, interpolate_xy=False) for path in paths]

    # Check that we're using Thomas's sanitised CSV files
    for db in dbs:
        assert not np.any(np.isnan(db.x))

    return dbs


@bobnav.cache_result
def _get_pm_headings(pm, test_df):
    return pm.ridf(test_df)


def to_merc(db):
    mlat, mlon = gm_plotting.utm_to_merc(db.x, db.y, 30, 'U')

    # Convert to x, y
    return mlon, mlat


def get_gps_quality(df):
    return df['GPS quality'].apply(pd.to_numeric, errors='coerce')


def run_analysis(
        train_route_path, test_route_paths, train_skips, test_skips, im_sizes,
        preprocess_strs, process_data=None):
    train_route = bobnav.Database(train_route_path)
    test_routes = [bobnav.Database(path) for path in test_route_paths]

    analysis_lst = []
    for train_skip in train_skips:
        for test_skip in test_skips:
            for im_size in im_sizes:
                for preprocess_str in preprocess_strs:
                    preprocess = (ip.resize(*im_size), eval(preprocess_str))
                    analysis = Analysis(train_route, train_skip, preprocess=preprocess)
                    for test_route in test_routes:
                        df = analysis.get_headings(test_route, test_skip)
                        if process_data:
                            process_data(train_route, test_route, df, train_skip, test_skip, im_size, preprocess)

                    analysis_lst.append(analysis)

    return analysis_lst


class Analysis:
    def __init__(self, train_route: bobnav.Database, train_skip, preprocess=None):
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
        test_df['heading_error'] = np.abs(bobnav.normalise180(np.rad2deg(dhead)))
        return test_df
