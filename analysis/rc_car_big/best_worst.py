# %%
import matplotlib.pyplot as plt
import rc_car_big
import numpy as np

import navbench as nb
from navbench import imgproc as ip
import bob_robotics.navigation as bobnav

TRAIN_SKIP = 10
TEST_SKIP = 40
PREPROC = ip.resize(180, 45)

paths = rc_car_big.get_paths()
dbs = rc_car_big.load_databases(paths[0:2])  #, limits_metres=(0, 200))

train_route = dbs[0]
test_route = dbs[1]

analysis = rc_car_big.Analysis(train_route, train_skip=TRAIN_SKIP, preprocess=PREPROC)
test_df = analysis.get_headings(test_route, TEST_SKIP)
test_df.sort_values('heading_error', inplace=True)

def get_data(images, snapshots, step=1):
    snapshots = bobnav.to_images_array(snapshots)
    pm = bobnav.PerfectMemory(snapshots[0].shape[::-1])
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
nearest_train_entry = analysis.train_route.read_image_entries(analysis.train_route.loc[datum.nearest_train_idx], preprocess=PREPROC)
target_snap = analysis.train_route.read_images(datum.nearest_train_idx, preprocess=PREPROC)
target_ridf = get_data(test_im, target_snap)

show_data(best_snap, datum, f'Best-matching snapshot (j={best_train_entry.database_idx})')
show_data(target_snap, target_ridf, f'Target snapshot (j={nearest_train_entry.database_idx})')

plt.show()

# %%
