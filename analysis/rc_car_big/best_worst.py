# %%
import matplotlib.pyplot as plt
import rc_car_big
import numpy as np
import gm_plotting

import navbench as nb
from navbench import imgproc as ip
import bob_robotics.navigation as bobnav

TRAIN_SKIP = 10
TEST_SKIP = 40
PREPROC = ip.resize(45, 180)
GOOD_GPS_ONLY = True

paths = rc_car_big.get_paths()
dbs = rc_car_big.load_databases(paths[0:2])  #, limits_metres=(0, 200))

train_route = dbs[0]
test_route = dbs[1]

analysis = rc_car_big.Analysis(train_route, train_skip=TRAIN_SKIP, preprocess=PREPROC)
test_df = analysis.get_headings(test_route, TEST_SKIP)
if GOOD_GPS_ONLY:
    gps = rc_car_big.get_gps_quality(test_df)
    test_df = test_df[rc_car_big.get_gps_quality(test_df) == 5]
test_df.sort_values('heading_error', inplace=True)

def distance2d(entry1, entry2):
    diff = np.array([entry1.x, entry1.y]) - np.array([entry2.x, entry2.y])
    if diff.ndim == 1:
        return np.hypot(*diff)
    else:
        return np.hypot(diff[0, :], diff[1, :])

def get_data(images, snapshots, step=1):
    snapshots = bobnav.to_images_array(snapshots)
    pm = bobnav.PerfectMemory(snapshots[0].shape)
    pm.train(snapshots)
    return pm.ridf(images, step=step)

def show_rot_diffim(ax, im, snap, theta, title):
    ax.cla()
    rim = np.roll(im, round(theta * im.shape[1] / (2 * np.pi)), axis=1)
    diffim = rim.astype(float) - snap.astype(float)
    diffim /= 255
    diffim = (diffim + 1) / 2
    show_im(ax, diffim, title, cmap=None)

def show_im(ax, im, title, cmap='gray'):
    ax.imshow(im, cmap=cmap)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

def show_data(snap, ridf, snap_title):
    _, ax = plt.subplots(4, 1)
    show_im(ax[0], test_im, f'Test image (i={datum.database_idx})')
    show_im(ax[1], snap, snap_title)
    nb.plot_ridf(ridf.ridf / 255, ax=ax[2], show_minimum=True)
    show_rot_diffim(ax[3], test_im, snap, ridf.estimated_dheading, "Difference")

# Show worst match
datum = test_df.iloc[-1]
print('Error: %.2f°' % datum.heading_error)

test_im = datum.image
best_train_entry = analysis.train_entries.loc[datum.best_snap_idx, :]
best_snap = best_train_entry.image
print(f'Best: {distance2d(datum, best_train_entry)}')

nearest_train_entry = analysis.train_route.read_image_entries(analysis.train_route.loc[datum.nearest_train_idx], preprocess=PREPROC)
target_snap = analysis.train_route.read_images(datum.nearest_train_idx, preprocess=PREPROC)
target_ridf = get_data(test_im, target_snap)
print(f'Target: {distance2d(datum, nearest_train_entry)}')

show_data(best_snap, datum, f'Best-matching snapshot (j={best_train_entry.database_idx})')
show_data(target_snap, target_ridf, f'Target snapshot (j={nearest_train_entry.database_idx})')

_, ax = plt.subplots()

test_x, test_y = rc_car_big.to_merc(datum)
best_x, best_y = rc_car_big.to_merc(best_train_entry)
target_x, target_y = rc_car_big.to_merc(nearest_train_entry)

ax.scatter(test_x, test_y, color='b')
ax.scatter(target_x, target_y, color='g')
ax.scatter(best_x, best_y, color='r')

assert best_train_entry.database_idx < nearest_train_entry.database_idx
entries = analysis.train_route.entries[best_train_entry.database_idx:nearest_train_entry.database_idx+1]
mx, my = rc_car_big.to_merc(entries)
ax.plot(mx, my)

all_x, all_y = rc_car_big.to_merc(test_df)
ax.set_xlim(all_x.min(), all_x.max())
ax.set_ylim(all_y.min(), all_y.max())

gm_plotting.APIClient().add_satellite_image_background(ax)

plt.show()

# %%
