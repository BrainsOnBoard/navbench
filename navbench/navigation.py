import numpy as np
import matplotlib.pyplot as plt


def mean_absdiff(x, y):
    """Return mean absolute difference between two images or sets of images
    (as 3D matrices). """

    # Check that the images are of the same size
    assert x.shape[0:2] == y.shape[0:2]

    # Must be floats (0 <= x <= 1)
    assert x.dtype == np.float and y.dtype == np.float

    # If the dimensions don't match up then we need to add an extra fake
    # dimension. Eww.
    if x.ndim == 2 and y.ndim == 3:
        x = x[:, :, np.newaxis]
    elif x.ndim == 3 and y.ndim == 2:
        y = y[:, :, np.newaxis]

    absdiffs = np.abs(x - y)
    return absdiffs.mean(axis=(0, 1))


def get_route_idf(images, snap):
    return mean_absdiff(images, snap)


def get_ridf(image, snap, step=1):
    assert step > 0

    nsteps = image.shape[1] // step
    cols = image.shape[2] if image.ndim == 3 else 1
    diffs = np.zeros([nsteps, cols], dtype=np.float)
    for i in range(nsteps):
        diffs[i, :] = mean_absdiff(image, snap)
        snap = np.roll(snap, -step, axis=1)

    return diffs


def normalise180(ths):
    for idx, th in enumerate(ths):
        if th > 180:
            ths[idx] -= 360
    return ths


def ridf_to_degrees(diffs):
    bestcols = np.argmin(diffs, axis=0)
    ths = 360 * bestcols / diffs.shape[0]
    return normalise180(ths)


def get_route_ridf(images, snap, step=1):
    return np.amin(get_ridf(images, snap, step), axis=0)


def get_route_ridf_headings(images, snap, step=1):
    diffs = get_ridf(images, snap, step)
    return ridf_to_degrees(diffs)


def plot_route_idf(entries, *diffs, labels=None):
    for i in range(len(diffs)):
        if labels:
            plt.plot(entries, diffs[i], label=labels[i])
        else:
            plt.plot(entries, diffs[i])
    plt.xlabel("Frame")
    plt.xlim(entries[0], entries[-1])
    plt.ylabel("Mean image diff (px)")
    plt.ylim(0, plt.ylim()[1])

    if labels:
        plt.legend()


def plot_ridf(diffs, ax=None):
    # We want the plot to go from -180° to 180°, so we wrap around
    diffs = np.append(diffs, diffs[0])
    diffs = np.roll(diffs, round(len(diffs) / 2))

    if ax is None:
        _, ax = plt.subplots()
    else:
        ax.clear()

    xs = np.linspace(-180, 180, len(diffs))
    ax.plot(xs, diffs)
    ax.set_xlim(-180, 180)
    ax.set_xticks(range(-180, 181, 45))

    return ax
