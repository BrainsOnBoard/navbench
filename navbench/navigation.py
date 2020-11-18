import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_3d_array(images):
    images = np.asarray(images)

    if images.ndim == 3:
        return images
    if images.ndim == 2:
        return images[np.newaxis, :, :]
    raise ValueError("images must be a 2d or 3d array")


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def mean_absdiff(x, y):
    """Return mean absolute difference between two images or sets of images."""

    if isinstance(x, list):
        x, y = y, x

    if isinstance(y, list):
        return [cv2.absdiff(x, im).mean() for im in y]

    return cv2.absdiff(x, y).mean()


def ridf(images, snap, step=1):
    assert step > 0
    assert step % 1 == 0

    num_images = len(images) if isinstance(images, list) else 1
    steps = range(0, snap.shape[1], step)
    diffs = np.empty((num_images, len(steps)), dtype=np.float)
    for i, rot in enumerate(steps):
        rsnap = np.roll(snap, -rot, axis=1)
        diffs[:, i] = mean_absdiff(images, rsnap)

    return diffs if diffs.shape[0] > 1 else diffs[0]

def route_idf(images, snap):
    return mean_absdiff(images, snap)


def normalise180(ths):
    for idx, th in enumerate(ths):
        if th > 180:
            ths[idx] -= 360
    return ths


def ridf_to_degrees(diffs):
    bestcols = np.argmin(diffs, axis=1)
    ths = 360 * bestcols / diffs.shape[1]
    return normalise180(ths)


def route_ridf(images, snap, step=1):
    return np.amin(ridf(images, snap, step), axis=1)


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


def plot_route_idf(entries, *errs_args, ax=None, filter_zeros=True, labels=None):
    if not labels:
        labels = len(errs_args[0]) * [None]

    if ax is None:
        _, ax = plt.subplots()

    for errs, label in zip(errs_args, labels):
        if filter_zeros:
            errs = zeros_to_nones(errs)
        ax.plot(entries, errs, label=label)

    ax.set_xlabel("Frame")
    ax.set_xlim(entries[0], entries[-1])
    ax.set_ylabel("Mean image diff (px)")
    ax.set_ylim(bottom=0)

    if filter_zeros:
        for errs in errs_args:
            for entry, val in zip(entries, errs):
                if val == 0:
                    ax.plot((entry, entry), ax.get_ylim(), 'r:')

    if labels[0]:
        ax.legend()


def plot_ridf(diffs, ax=None, im=None):
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
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(-180, 181, 45))

    if im is not None:
        ext = (*ax.get_xlim(), *ax.get_ylim())
        ax.imshow(im, cmap='gray', zorder=0, extent=ext)
        ax.set_aspect((im.shape[0] / im.shape[1]) *
                      ((ext[1] - ext[0]) / (ext[3] - ext[2])))

    return ax
