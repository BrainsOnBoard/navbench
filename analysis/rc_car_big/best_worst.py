import matplotlib.pyplot as plt
import rc_car_big

import navbench as nb
from navbench import imgproc as ip

TRAIN_SKIP = 40
TEST_SKIP = 1000
PREPROC = ip.resize(180, 45)

paths = rc_car_big.get_paths()
dbs = rc_car_big.load_databases(paths[0:2])  #, limits_metres=(0, 200))

train_route = dbs[0]
test_route = dbs[1]

analysis = rc_car_big.Analysis(train_route, train_skip=TRAIN_SKIP, preprocess=PREPROC)
test_df = analysis.get_headings(test_route, TEST_SKIP)
test_df.sort_values('heading_error', inplace=True)

def show_im(ax, im, title):
    ax.imshow(im, cmap='gray')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

def plot_test_point(datum):
    datum = datum.squeeze()
    print('Error: %.2f°' % datum.heading_error)

    _, ax = plt.subplots(5, 1)
    test_im = datum.image
    best_train_entry = analysis.train_entries.loc[datum.best_snap_idx, :]
    train_im = best_train_entry.image
    nearest_train_entry = analysis.train_route.read_image_entries(analysis.train_route.loc[datum.nearest_train_idx], preprocess=PREPROC, to_float=False)
    target_snap = analysis.train_route.read_images(datum.nearest_train_idx, preprocess=PREPROC, to_float=False)

    show_im(ax[0], test_im, 'Test image (%.2f m)' % nearest_train_entry.distance)
    show_im(ax[1], train_im, 'Best-matching snapshot (%.2f m)' % best_train_entry.distance)
    nb.plot_ridf(datum.ridf / 255, ax=ax[2], show_minimum=True)
    ax[2].set_title(f'RIDF (snapshot={datum.best_snap_idx})')
    show_im(ax[3], target_snap, '"Target" snapshot')
    target_ridf = analysis.pm.ridf(nearest_train_entry)
    nb.plot_ridf(target_ridf.ridf / 255, ax=ax[4], show_minimum=True)
    ax[4].set_title(f'RIDF (snapshot={nearest_train_entry.database_idx})')

# Show worst match
plot_test_point(test_df.iloc[-1])
plt.show()
