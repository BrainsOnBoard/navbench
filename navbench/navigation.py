import numpy as np
import matplotlib.pyplot as plt


def to_3d_array(images):
    images = np.asarray(images)

    if images.ndim == 3:
        return images
    if images.ndim == 2:
        return images[np.newaxis, :, :]
    raise ValueError("images must be a 2d or 3d array")


def mean_absdiff(x, y):
    """Return mean absolute difference between two images or sets of images
    (as 3D matrices). """
    x = to_3d_array(x).astype(np.float)
    y = to_3d_array(y).astype(np.float)

    # Check that the images are of the same size
    assert x.shape[1:] == y.shape[1:]

    absdiffs = np.abs(x - y)
    return absdiffs.mean(axis=(1, 2))


def ridf(images, snap, step=1):
    assert step > 0
    assert step % 1 == 0
    images = to_3d_array(images)
    snap = to_3d_array(snap)

    steps = range(0, images.shape[2], step)
    diffs = np.empty((len(images), len(steps)), dtype=np.float)
    for i, rot in enumerate(steps):
        rsnap = np.roll(snap, -rot, axis=2)
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


def plot_route_idf(entries, *diffs, labels=None):
    for diff, label in zip(diffs, labels):
        if labels:
            plt.plot(entries, diff, label=label)
        else:
            plt.plot(entries, diff)
    plt.xlabel("Frame")
    plt.xlim(entries[0], entries[-1])
    plt.ylabel("Mean image diff (px)")
    plt.ylim(0, plt.ylim()[1])

    if labels:
        plt.legend()


def plot_ridf(diffs, ax=None):
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

    return ax
