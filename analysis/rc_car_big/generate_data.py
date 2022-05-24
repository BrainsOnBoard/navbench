# From todo list dated 2022-05-24
import rc_car_big

TRAIN_SKIP = [5, 10, 20]
TEST_SKIP = [10]
TEST_SKIP_OFFSET = range(TEST_SKIP[0])
IM_SIZE = [(45, 180)]
PREPROC = ['ip.remove_sky_and_histeq']

paths = rc_car_big.get_paths()
rc_car_big.run_analysis(paths, paths, TRAIN_SKIP, TEST_SKIP, IM_SIZE, PREPROC, TEST_SKIP_OFFSET, export_mats=True)
