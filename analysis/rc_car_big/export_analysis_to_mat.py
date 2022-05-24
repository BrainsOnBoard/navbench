# %%
import rc_car_big
import bob_robotics.navigation as bobnav

TRAIN_SKIP = [10]
TEST_SKIP = [80]
# IM_SIZE = [(45, 180)]
IM_SIZE = [(360, 1440), (180, 720), (90, 360), (45, 180), (22, 90)]
PREPROC = ['None', 'ip.histeq']

paths = rc_car_big.get_paths()
rc_car_big.run_analysis(
    [paths[0]],
    [paths[1]],
    TRAIN_SKIP,
    TEST_SKIP,
    IM_SIZE,
    PREPROC,
    do_export_mats=True)
