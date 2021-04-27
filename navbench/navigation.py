import cv2
import numpy as np
import matplotlib.pyplot as plt


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

    return __ridf(images, snapshots, difference, step)


def get_ridf_headings(images, snapshots, step=1):
    heads = []
    for image in images:
        diffs = ridf(image, snapshots, step=step)
        best_over_rot = np.min(diffs, axis=1)
        best_row = np.argmin(best_over_rot)
        heads.append(ridf_to_radians(diffs[best_row, :]))
    return np.array(heads)


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
