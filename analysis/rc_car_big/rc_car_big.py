import os
import sqlite3
import subprocess as sp
from glob import glob
from math import gcd
from shutil import make_archive
from time import perf_counter
from typing import *
from urllib.parse import urlencode
from warnings import warn

import bob_robotics.navigation as bobnav
import gm_plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bob_robotics.navigation import imgproc as ip
from paramspace import ParamDim, ParamSpace
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

    # This column contains a huuuuge amount of data, so let's do without it.
    # Otherwise the cache folder will end up massive.
    df.drop('differences', axis=1, inplace=True)

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
        'params': params,
        'git_commit': _get_git_commit(),
        'bob_robotics_version': bobnav.__version__,
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
        preprocess_strs, test_skip_offsets=(0,), runner_classes=None, show_plots=False,
        export_mats=False):
    train_routes = [bobnav.Database(path) for path in train_route_paths]
    test_routes = [bobnav.Database(path) for path in test_route_paths]

    if not runner_classes:
        runner_classes = []
    if export_mats:
        runner_classes.append(ExportMatFilesRunner)

    for offset in test_skip_offsets:
        assert offset >= 0

        for train_skip in train_skips:
            for test_skip in test_skips:
                assert offset < test_skip

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
                                df = analysis.get_headings(test_route, test_skip, offset)
                                for runner in runners:
                                    runner.on_test(train_route, test_route, df, preprocess, {
                                        'database_name': test_route.name, **params})

                            for runner in runners:
                                runner.on_finish(params)

                            if show_plots:
                                plt.show()


def _path_list(paths):
    if isinstance(paths, str):
        paths = [paths]

    return paths

class DatabaseCache:
    _table_name = 'images'

    def __init__(self, paths):
        self.dbs = dict()
        for path in paths:
            if not path in self.dbs:
                self.dbs[path] = bobnav.Database(path)

        self.con = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'database_cache.db'))
        self.cur = self.con.cursor()

        self.cur.execute(f'''CREATE TABLE IF NOT EXISTS {self._table_name}
                         (id integer primary key, database_name, database_idx, im_height, im_width,
                         preprocess, image)''')

    def get_entries(self, db, indexes, im_size, preprocess_str):
        if isinstance(indexes, Callable):
            indexes = indexes(len(db))
        indexes = list(indexes)

        # Load any missing entries and save to cache
        existing_idx = self.cur.execute(f'''SELECT database_idx FROM {self._table_name}
            WHERE database_name = ?
            AND im_height = ? AND im_width = ? AND preprocess = ?
            AND database_idx IN ({','.join(map(str, indexes))})''', (db.name,
            *im_size, preprocess_str)).fetchall()
        not_in_db = set(indexes) - set(x[0] for x in existing_idx)
        print(f'query {db.name} ({len(existing_idx)} / {len(not_in_db)})')
        if not_in_db:
            preprocess = (ip.resize(*im_size), eval(preprocess_str))
            df2 = db.read_image_entries(list(not_in_db), preprocess=preprocess)
            df2 = df2[['database_idx', 'image']]
            df2['database_name'] = db.name
            df2['im_height'] = im_size[0]
            df2['im_width'] = im_size[1]
            df2['preprocess'] = preprocess_str

            df2.to_sql(self._table_name, self.con, if_exists='append', index=False)

        # TODO: Better way to express indexes?
        stmt = f'''SELECT * FROM {self._table_name}
            WHERE database_name = ?
            AND im_height = ? AND im_width = ? AND preprocess = ?
            AND database_idx IN ({','.join(map(str, indexes))})'''

        # TODO: Allow for users setting chunksize here?
        df = pd.read_sql_query(stmt, self.con, params=(db.name, *im_size, preprocess_str))
        if len(df) == 0:
            print(stmt)

        # TODO: Handle zero-length case?
        df.image = df[['im_height', 'im_width', 'image']].apply(
            lambda row: np.frombuffer(row.image, dtype=np.uint8).reshape(row.im_height, row.im_width), axis=1)

        # YUCK: rsuffix is needed but I don't know why. Why can't it merge the columns?
        df = df.join(db.entries.loc[indexes], on='database_idx', rsuffix='_db', sort=True)
        return df

    def __del__(self):
        self.con.close()

class PMResultsCache:
    _table_name = 'results'

    def __init__(self, db_cache : DatabaseCache):
        self.con = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'results_cache.db'))
        self.cur = self.con.cursor()
        self.db_cache = db_cache

    def __del__(self):
        self.con.close()

    def get_data(
            self, train_route_path, test_route_path, train_skip, test_skip,
            test_skip_offset, im_size, preprocess_str):

        train_route = self.db_cache.dbs[train_route_path]
        # train_indexes = range(0, len(train_route))
        # !!!!!!!!!!!!!!! FIX ME
        train_indexes = range(10)
        train_df = self.db_cache.get_entries(train_route, train_indexes, im_size, preprocess_str)

        pm = bobnav.PerfectMemory(train_images=train_df)

        test_route = self.db_cache.dbs[test_route_path]
        test_indexes = [(test_skip_offset + x) for x in range(0, len(test_route), test_skip)]
        test_df = self.db_cache.get_entries(test_route, test_indexes, im_size, preprocess_str)
        # return pm.ridf(test_df)
        diffs = pm.ridf_raw(test_df)
        print('GOT RAW DIFFS')
        test_df['differences'] = diffs
        test_df2 = pm.ridf(test_df)
        return test_df2

def run_analysis_params(pspace, to_run=()):
    train_route_paths = _path_list(pspace['train_route_path'])
    test_route_paths = _path_list(pspace['test_route_path'])
    db_cache = DatabaseCache(train_route_paths + test_route_paths)
    results_cache = PMResultsCache(db_cache)

    for params in ParamSpace(pspace):
        data = results_cache.get_data(**params)
        train_route = db_cache.dbs[params['train_route_path']]
        test_route = db_cache.dbs[params['test_route_path']]
        for fun in to_run:
            fun(data, train_route=train_route, test_route=test_route, **params)

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

    def get_headings(self, test_route, test_skip, test_skip_offset=0):
        test_df = test_route.read_image_entries(
            test_route.iloc[test_skip_offset::test_skip],
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
