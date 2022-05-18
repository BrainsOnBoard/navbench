# %%
import os

import rc_car_big
from bob_robotics.navigation import imgproc as ip
import bob_robotics.navigation as bobnav
from scipy.io import savemat
from shutil import make_archive

MY_PATH = os.path.dirname(__file__)
TRAIN_SKIP = 1
TEST_SKIP = 80
IM_SIZE = (45, 180)
PREPROC = 'None'

MAT_ROOT = os.path.join(MY_PATH, 'mat_files')
MAT_SUFFIX = f'data_train_skip={TRAIN_SKIP}_test_skip={TEST_SKIP}_imsize={IM_SIZE[0]}x{IM_SIZE[1]}_preproc={PREPROC}'
MAT_PATH = os.path.join(MAT_ROOT, MAT_SUFFIX)

try:
    os.mkdir(MAT_PATH)
except FileExistsError:
    pass


def filter_col_name(name: str):
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
    dict_out['preprocessing'] = PREPROC

    filepath = os.path.join(MAT_PATH, filename)
    assert not os.path.exists(filepath)
    savemat(filepath, dict_out, appendmat=False, oned_as='column')


def save_test_data(analysis, test_route, df):
    # This column contains a huuuuge amount of data, so let's do without it.
    # (Removing it decreased the size of my .mat file from >600MB to <1MB.)
    df.drop('differences', axis=1, inplace=True)

    # Save RIDF for nearest point on training route too
    train_images = analysis.train_route.read_images(
        df.nearest_train_idx.to_list(),
        preprocess=analysis.preprocess)
    nearest_ridfs = []
    for image, snap in zip(df.image, train_images):
        nearest_ridfs.append(bobnav.ridf(image, snap))
    df['nearest_ridf'] = nearest_ridfs

    # These are possibly confusing
    df.drop(columns=['yaw', 'best_snap'], axis=1, inplace=True)

    save_df(f'test_{test_route.name}.mat', df, test_route)


paths = rc_car_big.get_paths()
analysis = rc_car_big.run_analysis(
    paths[0],
    [paths[1]],
    TRAIN_SKIP, TEST_SKIP, IM_SIZE, PREPROC, save_test_data)
save_df('train.mat', analysis.train_entries, analysis.train_route)

make_archive(MAT_SUFFIX, 'zip', root_dir=MAT_ROOT, base_dir=MAT_SUFFIX)
