import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    import pathos.multiprocessing as mp
except:
    mp = None
    print('WARNING: Could not find pathos.multiprocessing module')

from . import caching


def mean_absdiff(x, y):
    """Return mean absolute difference between two images or sets of images."""

    if isinstance(x, list):
        x, y = y, x

    if isinstance(y, list):
        return [cv2.absdiff(x, im).mean() for im in y]

    return cv2.absdiff(x, y).mean()


def rotate_pano(image, right_deg):
    rot_px = int(right_deg * image.shape[1] / 360)
    return np.roll(image, rot_px, axis=1)


def __ridf(test_images, ref_image, difference, step):
    """Internal function; do not use directly"""
    step_max = ref_image.shape[1]
    if step < 0:
        step_max = -step_max
    steps = range(0, step_max, step)

    diffs = np.empty((len(test_images), len(steps)), dtype=float)
    for i, rot in enumerate(steps):
        rref = np.roll(ref_image, -rot, axis=1)
        diffs[:, i] = difference(test_images, rref)

    return diffs if diffs.shape[0] > 1 else diffs[0]


def ridf(images, snapshots, difference=mean_absdiff, step=1):
    """Return an RIDF for one or more images vs one or more snapshots (as a vector)"""
    assert step > 0
    assert step % 1 == 0

    multi_images = isinstance(images, list)
    multi_snaps = isinstance(snapshots, list)
    assert not multi_images or not multi_snaps
    if multi_snaps:
        return __ridf(snapshots, images, difference, -step)

    if not multi_images:
        images = [images]

    return __ridf(images, snapshots, difference, step)


def get_ridf_headings_no_cache(images, snapshots, step=1, parallel=None):
    """A version of get_ridf_headings() without on-disk caching of results.

    Parameters are the same as for get_ridf_headings().
    """
    if len(images) == 0 or len(snapshots) == 0:
        return np.array(())

    # Get a heading for a single image
    def get_heading_for_image(image):
        diffs = ridf(image, snapshots, step=step)
        if isinstance(snapshots, list):
            best_over_rot = np.min(diffs, axis=1)
            best_row = np.argmin(best_over_rot)
            diffs = diffs[best_row, :]
        return ridf_to_radians(diffs)

    def run_serial():
        return np.array([get_heading_for_image(image) for image in images])

    def run_parallel():
        with mp.Pool() as pool:
            return np.array(pool.map(get_heading_for_image, images))

    if parallel is None:
        # Module not installed
        if not mp:
            return run_serial()

        # Process in parallel if we have the module and there is a fair
        # amount of processing to be done
        num_ops = len(images) * len(snapshots) * images[0].size

        # This value was determined quasi-experimentally on my home machine -- AD
        if num_ops >= 120000:
            return run_parallel()
        return run_serial()

    if parallel:
        if mp:
            return run_parallel()

        print('WARNING: Parallel processing requested but pathos.multiprocessing module is not available')

    return run_serial()


@caching.cache_result
def get_ridf_headings(images, snapshots, step=1, parallel=None):
    """Get a numpy array of headings computed from multiple images and snapshots

    Parameters
    ----------
    images
        List of test images
    snapshots
        List of testing images
    step
        Step size for each rotation in pixels
    parallel
        Whether to run algorithm in parallel. If omitted, it will be run in
        parallel according to a heuristic (i.e. if there is enough work)

    Returns:
        Headings in radians.
    """
    return get_ridf_headings_no_cache(images, snapshots, step, parallel)


def route_idf(images, snap):
    return mean_absdiff(images, snap)


def normalise180(ths):
    if np.isscalar(ths):
        return normalise180([ths])[0]

    ths = np.array(ths) % 360
    ths[ths > 180] -= 360
    return ths


def ridf_to_degrees(diffs):
    assert diffs.ndim == 1 or diffs.ndim == 2
    bestcols = np.argmin(diffs, axis=-1)
    return 360.0 * bestcols / diffs.shape[-1]


def ridf_to_radians(diffs):
    return np.deg2rad(ridf_to_degrees(diffs))


def route_ridf(images, snap, step=1):
    return np.amin(ridf(images, snap, step=step), axis=1)


def route_ridf_errors(images, snap, step=1):
    diffs = ridf(images, snap, step)
    return [abs(th) for th in ridf_to_degrees(diffs)]


def zeros_to_nones(vals):
    zeros = 0
    ret = []
    for val in vals:
        if val == 0:
            ret.append(None)
            zeros += 1
        else:
            ret.append(val)

    if zeros > 0:
        print('%i zero values (perfect matches?) are not being shown' % zeros)

    return ret


def plot_route_idf(xs, *errs_args, ax=None, filter_zeros=True, xlabel='Frame',
                   labels=None, adjust_ylim=True):
    if not labels:
        labels = len(errs_args[0]) * [None]

    if ax is None:
        _, ax = plt.subplots()

    lines = []
    for errs, label in zip(errs_args, labels):
        if filter_zeros:
            errs = zeros_to_nones(errs)
        lines.append(ax.plot(xs, errs, label=label)[0])

    ax.set_xlabel(xlabel)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylabel("Mean image diff (px)")
    if adjust_ylim:
        ax.set_ylim(bottom=0)

    if filter_zeros:
        for errs in errs_args:
            for entry, val in zip(xs, errs):
                if val == 0:
                    ax.plot((entry, entry), ax.get_ylim(), 'r:')

    if labels[0]:
        ax.legend()

    return lines


def plot_ridf(diffs, ax=None, im=None, adjust_ylim=True, show_minimum=False):
    assert diffs.ndim == 1

    # We want the plot to go from -180° to 180°, so we wrap around
    diffs = np.roll(diffs, round(len(diffs) / 2))
    diffs = np.append(diffs, diffs[0])

    if ax is None:
        _, ax = plt.subplots()
    else:
        ax.clear()

    xs = np.linspace(-180, 180, len(diffs))
    ax.plot(xs, diffs)
    ax.set_xlim(-180, 180)
    if adjust_ylim:
        ax.set_ylim(bottom=0)
    ax.set_xticks(range(-180, 181, 45))

    if show_minimum:
        idx = np.argmin(diffs)
        ax.plot([xs[idx]] * 2, ax.get_ylim(), 'k--', alpha=0.7)

    if im is not None:
        ext = (*ax.get_xlim(), *ax.get_ylim())
        ax.imshow(im, cmap='gray', zorder=0, extent=ext)
        ax.set_aspect((im.shape[0] / im.shape[1]) *
                      ((ext[1] - ext[0]) / (ext[3] - ext[2])))

    return ax
