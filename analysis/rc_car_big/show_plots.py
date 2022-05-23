# %%

from plots import *

import rc_car_big

PLOTS = [RouteQuiver, DistanceVsError]

TRAIN_SKIP = 10
TEST_SKIP = 80
IM_SIZE = (45, 180)
PREPROC = 'None'

paths = rc_car_big.get_paths()


rc_car_big.run_analysis(
    paths[0:1],
    paths[1:],
    [TRAIN_SKIP],
    [TEST_SKIP],
    [IM_SIZE],
    [PREPROC],
    runner_classes=PLOTS,
    show_plots=True)
