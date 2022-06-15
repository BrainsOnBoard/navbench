import bob_robotics.navigation as bobnav
from bob_robotics.navigation import imgproc as ip
import rc_car_big

IM_SIZE = (45, 180)
PREPROC = ip.resize(*IM_SIZE)

paths = rc_car_big.get_paths()
train_route = bobnav.Database(paths[0])
test_route = bobnav.Database(paths[1])

train_entries = train_route.read_image_entries(preprocess=PREPROC)
test_entries = test_route.read_image_entries(preprocess=PREPROC)

print(f'Doing {len(train_entries) * len(test_entries)} comparisons')
pm = bobnav.PerfectMemory(IM_SIZE)
pm.train(train_entries)
df = pm.ridf(test_entries)
