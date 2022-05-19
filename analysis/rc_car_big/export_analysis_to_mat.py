# %%
import os

import rc_car_big
import bob_robotics.navigation as bobnav
from scipy.io import savemat
from shutil import make_archive
from urllib.parse import urlencode

TRAIN_SKIP = [10]
TEST_SKIP = [80]
IM_SIZE = [(45, 180)]
PREPROC = ['None']
OVERWRITE = True

MAT_ROOT = os.path.join(os.path.dirname(__file__), 'mat_files')


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

    dict_out = {'params': params, **df_out.to_dict('list')}

    # Make folder, deriving its name from parameter values
    mat_path = os.path.join(MAT_ROOT, get_mat_folder_name(params))
    try:
        os.mkdir(mat_path)
    except FileExistsError:
        pass

    filepath = os.path.join(mat_path, filename)
    assert OVERWRITE or not os.path.exists(filepath)
    savemat(filepath, dict_out, appendmat=False, oned_as='column')


def save_train_data(analysis, preprocess, params):
    save_df('train.mat', analysis.train_entries, params)


def save_test_data(train_route, test_route, df, preprocess, params):
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


def archive_data(params):
    mat_suffix = get_mat_folder_name(params)
    make_archive(mat_suffix, 'zip', root_dir=MAT_ROOT, base_dir=mat_suffix)


paths = rc_car_big.get_paths()
rc_car_big.run_analysis(
    paths[0],
    [paths[1]],
    TRAIN_SKIP,
    TEST_SKIP,
    IM_SIZE,
    PREPROC,
    train_hook=save_train_data,
    test_hook=save_test_data,
    post_test_hook=archive_data)
