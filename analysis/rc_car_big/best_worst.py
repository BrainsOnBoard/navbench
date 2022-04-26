# %%
import matplotlib.pyplot as plt
import rc_car_big
import numpy as np

import navbench as nb
from navbench import imgproc as ip

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

def show_rot_diffim(ax, im, snap, theta, title):
    rim = np.roll(im, round(theta * im.shape[1] / (2 * np.pi)), axis=1)
    diffim = rim.astype(float) - snap.astype(float)
    diffim /= 255
    diffim = (diffim + 1) / 2
    show_im(ax, diffim, title, cmap='hot')

def show_im(ax, im, title, cmap='gray'):
    ax.imshow(im, cmap=cmap)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

# Show worst match
datum = test_df.iloc[-1]
datum = datum.squeeze()
print('Error: %.2f°' % datum.heading_error)

_, ax0 = plt.subplots(4, 1)
test_im = datum.image
best_train_entry = analysis.train_entries.loc[datum.best_snap_idx, :]
best_snap = best_train_entry.image
nearest_train_entry = analysis.train_route.read_image_entries(analysis.train_route.loc[datum.nearest_train_idx], preprocess=PREPROC, to_float=False)

show_im(ax0[0], test_im, 'Test image (%.2f m, i=%d)' % (nearest_train_entry.distance, datum.database_idx))
show_im(ax0[1], best_snap, 'Best-matching snapshot (%.2f m, i=%d)' % (best_train_entry.distance, best_train_entry.database_idx))
nb.plot_ridf(datum.ridf / 255, ax=ax0[2], show_minimum=True)
ax0[2].set_title(f'RIDF (snapshot={datum.best_snap_idx})')
show_rot_diffim(ax0[3], test_im, best_snap, datum.estimated_dheading, "Difference")

_, ax1 = plt.subplots(4, 1)
show_im(ax1[0], test_im, 'Test image (%.2f m)' % nearest_train_entry.distance)
target_snap = analysis.train_route.read_images(datum.nearest_train_idx, preprocess=PREPROC, to_float=False)
show_im(ax1[1], target_snap, '"Target" snapshot')
target_ridf = analysis.pm.ridf(nearest_train_entry)
nb.plot_ridf(target_ridf.ridf / 255, ax=ax1[2], show_minimum=True)
ax1[2].set_title(f'RIDF (snapshot={nearest_train_entry.database_idx})')
show_rot_diffim(ax1[3], test_im, target_snap, target_ridf.estimated_dheading, "Difference")
plt.show()

# %%
