from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import bob_robotics.navigation as bobnav

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
    if hasattr(x, "to_list"):
        # Convert elements of DataFrame. The .to_numpy() method doesn't work
        # here because our data are multi-dimensional arrays.
        x = x.to_list()

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


def ridf(images, snapshots, step=1):
    """Return an RIDF for one or more images vs one or more snapshots (as a vector)"""
    snapshots = to_images_array(snapshots)
    pm = bobnav.PerfectMemory(snapshots[0].shape[::-1])
    pm.train(snapshots)
    return np.array(pm.ridf(images, step=step).ridf.to_list())


def get_ridf_headings_no_cache(images, snapshots, step=1):
    """A version of get_ridf_headings() without on-disk caching of results.

    Parameters are the same as for get_ridf_headings().
    """
    snapshots = to_images_array(snapshots)
    pm = bobnav.PerfectMemory(snapshots[0].shape[::-1])
    pm.train(snapshots)
    return pm.ridf(images, step=step).estimated_heading


@caching.cache_result
def get_ridf_headings(images, snapshots, step=1):
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
    return get_ridf_headings_no_cache(images, snapshots, step)


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
    diffs = ridf(images, snap, step=step)
    return np.abs(normalise180(ridf_to_degrees(diffs)))


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
