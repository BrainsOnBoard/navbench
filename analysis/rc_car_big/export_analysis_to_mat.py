# %%
import os

import rc_car_big
from bob_robotics.navigation import imgproc as ip
from scipy.io import savemat
import numpy as np

MY_PATH = os.path.dirname(__file__)
TRAIN_SKIP = 1
TEST_SKIP = 80
# PREPROC = None
PREPROC = ip.resize(45, 180)
MAT_PATH = os.path.join(MY_PATH, 'mat_files')

try:
    os.mkdir(MAT_PATH)
except FileExistsError:
    pass


paths = rc_car_big.get_paths()
dbs = rc_car_big.load_databases(paths[0:2])  #, limits_metres=(0, 200))

train_route = dbs[0]
test_routes = dbs[1:]
analysis = rc_car_big.Analysis(train_route, train_skip=TRAIN_SKIP, preprocess=PREPROC)

def filter_col_name(name : str):
    return name.replace("[", "").replace("]", "").replace(" ", "_")

def save_df(filename, df, db):
    df_out = df.rename(columns=filter_col_name)

    # Don't give the absolute path, as that's probably not useful
    df_out.filepath = df_out.filepath.apply(os.path.basename)

    # Indexes start at 1 in Matlab
    idx_cols = (col for col in df_out.columns if col.endswith("_idx"))
    for col in idx_cols:
        df_out[col] += 1

    dict_out = df_out.to_dict('list')
    dict_out['database_name'] = db.name
    dict_out['preprocessing'] = repr(PREPROC)

    filepath = os.path.join(MAT_PATH, filename)
    assert not os.path.exists(filepath)
    savemat(filepath, dict_out, appendmat=False, oned_as='column')

save_df('train.mat', analysis.train_entries, train_route)

for test_route in test_routes:
    df = analysis.get_headings(test_route, TEST_SKIP)

    # This column contains a huuuuge amount of data, so let's do without it.
    # (Removing it decreased the size of my .mat file from >600MB to <1MB.)
    df.drop('differences', axis=1, inplace=True)

    # These are possibly confusing
    df.drop(columns=['yaw', 'best_snap'], axis=1, inplace=True)

    save_df(f'test_{test_route.name}.mat', df, test_route)
