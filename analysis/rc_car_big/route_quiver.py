# %%
import rc_car_big
import matplotlib.pyplot as plt
import numpy as np
import navbench as nb

TRAIN_SKIP = 40
TEST_SKIP = 40
TO_FLOAT = False

def get_valid_entries(db, skip):
    lst = [i for i, x in enumerate(db.x) if not np.isnan(x)]
    return lst[::skip]

paths = rc_car_big.get_paths()
dbs = rc_car_big.load_databases(paths[0:6], limits_metres=(0, 100))

train_route = dbs[0]
test_routes = dbs[1:]
heading_offset = train_route.calculate_heading_offset(0.25)

train_entries = get_valid_entries(train_route, TRAIN_SKIP)
train_images = train_route.read_images(train_entries, to_float=TO_FLOAT)
print(f'Training images: {len(train_images)}')

_, ax = plt.subplots()
ax.plot(train_route.x, train_route.y)

# TODO: Could do e.g. medianfilt over these headings
gps_headings = np.arctan2(np.diff(train_route.y), np.diff(train_route.x))
gps_headings = np.append(gps_headings, [gps_headings[-1]])
snapshot_headings = gps_headings[train_entries]

for test_route in test_routes:
    test_entries = get_valid_entries(test_route, TEST_SKIP)
    test_images = test_route.read_images(test_entries, to_float=TO_FLOAT)

    print(f'Test images: {len(test_images)}')

    headings, best_snaps = nb.get_ridf_headings_and_snap(test_images, train_images)
    headings += snapshot_headings[best_snaps]

    lines = ax.plot(test_route.x, test_route.y, '--', label=test_route.name.replace('unwrapped_',''))
    colour = lines[0].get_color()
    ax.plot(test_route.x[0], test_route.y[0], 'o', color=colour)

    nb.anglequiver(
        ax,
        test_route.x[test_entries],
        test_route.y[test_entries],
        headings, color=colour, zorder=float('inf'), scale=300, scale_units='xy',
        alpha=0.8)

ax.legend()
plt.axis('equal')
plt.show()
