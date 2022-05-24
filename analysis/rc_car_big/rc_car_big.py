import os
from glob import glob
from shutil import make_archive
import subprocess as sp
from time import perf_counter
from urllib.parse import urlencode
from warnings import warn

import bob_robotics.navigation as bobnav
import gm_plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bob_robotics.navigation import imgproc as ip
from scipy.io import savemat

ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
DBROOT = os.path.join(ROOT, 'datasets/rc_car/rc_car_big')
MAT_ROOT = os.path.join(os.path.dirname(__file__), 'mat_files')
OVERWRITE_MATS = True

_git_commit = None


def _get_git_commit():
    global _git_commit
    if _git_commit:
        return _git_commit

    mydir = os.path.dirname(__file__)

    # Try to make a version string based on status of git tree
    try:
        output = sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=mydir)
        _git_commit = output.decode('utf-8').rstrip()

        ret = sp.run(["git", "diff", "--no-ext-diff",
                     "--quiet", "--exit-code"], cwd=mydir)
        if ret.returncode != 0:
            _git_commit += '.dirty'

    except sp.CalledProcessError:
        warn('Could not get current git commit; version of code is unknown')
        _git_commit = 'UNKNOWN'

    return _git_commit


def get_paths():
    return glob(os.path.join(DBROOT, 'unwrapped_*'))


def load_databases(paths=get_paths(), limits_metres=None):
    dbs = [
        bobnav.Database(
            path, limits_metres=limits_metres, interpolate_xy=False)
        for path in paths]

    # Check that we're using Thomas's sanitised CSV files
    for db in dbs:
        assert not np.any(np.isnan(db.x))

    return dbs


@bobnav.cache_result
def _get_pm_headings(pm, test_df):
    t0 = perf_counter()
    df = pm.ridf(test_df)
    df['computation_time'] = perf_counter() - t0
    return df


def to_merc(db):
    mlat, mlon = gm_plotting.utm_to_merc(db.x, db.y, 30, 'U')

    # Convert to x, y
    return mlon, mlat


def get_gps_quality(df):
    return df['GPS quality'].apply(pd.to_numeric, errors='coerce')


def filter_col_name(name: str):
    return name.replace("[", "").replace("]", "").replace(" ", "_")


def get_mat_folder_name(params):
    if 'database_name' in params:
        params = params.copy()
        del params['database_name']

    params_ordered = sorted(params.items(), key=lambda val: val[0])
    return urlencode(params_ordered, doseq=True).replace("&", "_")


def save_df(filename, df, params):
    df_out = df.rename(columns=filter_col_name)

    # Don't give the absolute path, as that's probably not useful
    df_out.filepath = df_out.filepath.apply(os.path.basename)

    # Indexes start at 1 in Matlab
    idx_cols = (col for col in df_out.columns if col.endswith("_idx"))
    for col in idx_cols:
        df_out[col] += 1

    dict_out = {
        'params': params, 'git_commit': _get_git_commit(),
        **df_out.to_dict('list')}

    # Make folder, deriving its name from parameter values
    mat_path = os.path.join(MAT_ROOT, get_mat_folder_name(params))
    try:
        os.mkdir(mat_path)
    except FileExistsError:
        pass

    filepath = os.path.join(mat_path, filename)
    assert OVERWRITE_MATS or not os.path.exists(filepath)
    savemat(filepath, dict_out, appendmat=False, oned_as='column')


class ExportMatFilesRunner:
    def __init__(self, analysis, preprocess, params):
        save_df('train.mat', analysis.train_entries, params)

    def on_test(self, train_route, test_route, df, preprocess, params):
        # This column contains a huuuuge amount of data, so let's do without it.
        # (Removing it decreased the size of my .mat file from >600MB to <1MB.)
        df.drop('differences', axis=1, inplace=True)

        # Save RIDF for nearest point on training route too
        train_images = train_route.read_images(
            df.nearest_train_idx.to_list(),
            preprocess=preprocess)
        nearest_ridfs = []
        for image, snap in zip(df.image, train_images):
            nearest_ridfs.append(bobnav.ridf(image, snap))
        df['nearest_ridf'] = nearest_ridfs

        # These are possibly confusing
        df.drop(columns=['yaw', 'best_snap'], axis=1, inplace=True)

        save_df(f'test_{test_route.name}.mat', df, params)

    def on_finish(self, params):
        mat_suffix = get_mat_folder_name(params)
        make_archive(mat_suffix, 'zip', root_dir=MAT_ROOT, base_dir=mat_suffix)


def run_analysis(
        train_route_paths, test_route_paths, train_skips, test_skips, im_sizes,
        preprocess_strs, runner_classes=None, show_plots=False,
        do_export_mats=False):
    train_routes = [bobnav.Database(path) for path in train_route_paths]
    test_routes = [bobnav.Database(path) for path in test_route_paths]

    if not runner_classes:
        runner_classes = []
    if do_export_mats:
        runner_classes.append(ExportMatFilesRunner)

    for train_skip in train_skips:
        for test_skip in test_skips:
            for im_size in im_sizes:
                for preprocess_str in preprocess_strs:
                    preprocess = (ip.resize(*im_size), eval(preprocess_str))

                    for train_route in train_routes:
                        analysis = Analysis(
                            train_route, train_skip, preprocess=preprocess)
                        params = {
                            'train_skip': train_skip, 'test_skip': test_skip,
                            'im_size': im_size, 'preprocess': preprocess_str}
                        runners = [
                            cur_class(
                                analysis, preprocess,
                                {'database_name': train_route.name, **params})
                            for cur_class in runner_classes]

                        for test_route in test_routes:
                            df = analysis.get_headings(test_route, test_skip)
                            for runner in runners:
                                runner.on_test(train_route, test_route, df, preprocess, {
                                    'database_name': test_route.name, **params})

                        for runner in runners:
                            runner.on_finish(params)

                        if show_plots:
                            plt.show()


class Analysis:
    def __init__(self, train_route: bobnav.Database, train_skip,
                 preprocess=None):
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
        test_df['heading_error'] = np.abs(
            bobnav.normalise180(np.rad2deg(dhead)))
        return test_df
