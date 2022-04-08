from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

try:
    import pathos.multiprocessing as mp
except:
    mp = None
    warn('Could not find pathos.multiprocessing module')

from . import caching


# Yes, there is np.atleast_3d, but the problem is that it tacks the extra
# dimension onto the end, whereas we want it to be at the beginning (e.g. you
# get an array of dims [90, 360, 1] rather than [1, 90, 360])
def to_images_array(x):
    try:
        # Convert elements of DataFrame. The .to_numpy() method doesn't work
        # here because our data are multi-dimensional arrays.
        x = x.to_list()
    except AttributeError:
        pass

    # Make sure x is a numpy array
    x = np.array(x)
    if x.ndim == 3:
        return x

    assert x.ndim == 2
    return np.array([x])


def mean_absdiff(x, y):
    """Return mean absolute difference between two images or sets of images."""
    x = to_images_array(x)
    y = to_images_array(y)

    # Either x or y can be 3D, but not both
    assert len(x) == 1 or len(y) == 1

    if len(x) > 1:
        x, y = y, x

    # Convert back to 2D
    x = np.squeeze(x, axis=0)

    ret = [cv2.absdiff(x, im).mean() for im in y]
    return ret if len(ret) > 1 else ret[0]


def rotate_pano(image, right_deg):
    rot_px = int(right_deg * image.shape[1] / 360)
    return np.roll(image, rot_px, axis=1)


def __ridf(test_images, ref_image, difference, step):
    """Internal function; do not use directly"""
    assert test_images.ndim == 3
    assert ref_image.ndim == 2

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
    images = to_images_array(images)
    snapshots = to_images_array(snapshots)
    if len(images) == 0 or len(snapshots) == 0:
        return []

    assert len(images) == 1 or len(snapshots) == 1

    if len(snapshots) > 1:
        return __ridf(snapshots, images[0], difference, -step)

    return __ridf(images, snapshots[0], difference, step)


def get_ridf_headings_no_cache(images, snapshots, step=1, parallel=None):
    """A version of get_ridf_headings() without on-disk caching of results.

    Parameters are the same as for get_ridf_headings().
    """
    images = to_images_array(images)
    snapshots = to_images_array(snapshots)

    # Get a heading for a single image
    def get_heading_for_image(image):
        diffs = ridf(image, snapshots, step=step)
        if len(snapshots) > 1:
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

        warn('Parallel processing requested but pathos.multiprocessing module is not available')

    return run_serial()

def get_ridf_headings_and_snap_no_cache(images, snapshots, step=1, parallel=None):
    """A version of get_ridf_headings() without on-disk caching of results.

    Parameters are the same as for get_ridf_headings().
    """

    # If images is from a DataFrame, keep the indexes around
    if hasattr(images, "iloc"):
        image_idx = images.index
    else:
        image_idx = range(len(images))
    if hasattr(snapshots, "iloc"):
        snap_idx = snapshots.index
    else:
        snap_idx = range(len(snapshots))

    images = to_images_array(images)
    snapshots = to_images_array(snapshots)

    def to_dataframe(seq):
        return DataFrame.from_records(seq, index=image_idx)

    # Get a heading for a single image
    def get_heading_for_image(image):
        diffs = ridf(image, snapshots, step=step)
        if len(snapshots) > 1:
            best_over_rot = np.min(diffs, axis=1)
            best_snap = np.argmin(best_over_rot)
            diffs = diffs[best_snap, :]
        else:
            best_snap = 0
        return {'estimated_heading': ridf_to_radians(diffs), 'best_snap': snap_idx[best_snap]}

    def run_serial():
        return to_dataframe(get_heading_for_image(image) for image in images)

    def run_parallel():
        with mp.Pool() as pool:
            return to_dataframe(pool.map(get_heading_for_image, images))

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

        warn('Parallel processing requested but pathos.multiprocessing module is not available')

    return run_serial()

@caching.cache_result
def get_ridf_headings_and_snap(*args, **kwargs):
    return get_ridf_headings_and_snap_no_cache(*args, **kwargs)

@caching.cache_result
def get_ridf_headings(images, snapshots, step=1, parallel=None):
    """Get a numpy array of headings computed from multiple images and snapshots

    Parameters
    ----------
    images
        List of test images
    snapshots
        List of training images
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
