# %%
import rc_car_big
import matplotlib.pyplot as plt
import numpy as np
import navbench as nb

TRAIN_SKIP = 40
TEST_SKIP = 40

paths = rc_car_big.get_paths()
dbs = rc_car_big.load_databases(paths) #[0:6], limits_metres=(0, 100))

train_route = dbs[0]
test_routes = dbs
analysis = rc_car_big.Analysis(train_route, TRAIN_SKIP)

for test_route in test_routes:
    analysis.get_headings(test_route, TEST_SKIP)
