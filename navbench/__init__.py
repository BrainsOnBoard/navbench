from collections.abc import Iterable
import math
from os import listdir
from os.path import isfile, join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import medfilt


def read_image_database(path):
    """Read info for image database entries from CSV file."""
    try:
        df = pd.read_csv(join(path, "database_entries.csv"))
    except FileNotFoundError:
        print("Warning: No CSV file found for", path)

        # If there's no CSV file then just treat all the files in the folder as
        # separate entries. Note that the files will be in alphabetical order,
        # which may not be what you want!
        entries = {
            "filepath": [f for f in [join(path, f) for f in listdir(path)] if isfile(f)]
        }
        entries["filepath"].sort()

        return entries

    # strip whitespace from column headers
    df = df.rename(columns=lambda x: x.strip())

    entries = {
        "x": df["X [mm]"] / 1000,
        "y": df["Y [mm]"] / 1000,
        "z": df["Z [mm]"] / 1000,
        "heading": df["Heading [degrees]"],
        "filepath": [join(path, fn.strip()) for fn in df["Filename"]]
    }

    return entries


def read_images(paths, preprocess=None):
    """Returns greyscale image(s) of type float."""

    # Single string as input
    if isinstance(paths, str):
        im = cv2.imread(paths, cv2.IMREAD_GRAYSCALE)

        # Run preprocessing step on images
        if preprocess:
            im = preprocess(im)

        return im

    if not paths:
        return []

    im0 = read_images(paths[0], preprocess)
    images = np.zeros([im0.shape[0], im0.shape[1], len(paths)], dtype=np.float)
    images[:, :, 0] = im0
    for i in range(1, len(paths)):
        images[:, :, i] = read_images(paths[i], preprocess)
    return images


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


def __get_ca_twoway(vals, filter_fun, thresh_fun, filter_size):
    if not vals:
        return 0

    # Must be numpy array
    vals = np.array(vals)

    # Assume goal is where minimum is
    goal = np.argmin(vals)

    def filter_vals(vec):
        vec = filter_fun(vec)

        # Median filtering doesn't make sense in this case
        if not vec.size:
            return np.empty(0)

        # This is invalid -- give error rather than spurious return values
        if len(vec) < filter_size:
            raise ValueError('Filter size is greater than vector length')

        return medfilt(vec, filter_size)

    # Apply filter to values from left and right of goal
    left = filter_vals(vals[goal::-1])
    right = filter_vals(vals[goal:])

    def get_ca(vec):
        if not vec.size:  # Empty array
            return 0

        return next(i for i, val in enumerate(vec) if thresh_fun(val))

    return get_ca(left) + get_ca(right)


def get_idf_ca(idf, filter_size=1):
    '''
    Get catchment area for 1D IDF.

    Differences from Andy's implementation:
        - I'm treating IDFs like this: [1, 2, 0, 2, 1] as having a CA of 2
          rather than 0.
        - IDFs which keep extending indefinitely to the left or right currently
          cause a ValueError to be thrown
        - Cases where vector length > filter size also cause an error
    '''
    try:
        return __get_ca_twoway(idf, np.diff, lambda x: x < 0, filter_size)
    except StopIteration:
        raise ValueError('IDF does not decrease at any point')


def get_rca(errs, thresh=45, filter_size=1):
    '''
    Get rotational catchment area:
        i.e., area over which abs(errs) < some_threshold

    Differences from Andy's implementation:
        - filter_size defaults to 1, not 3
    '''
    assert thresh >= 0

    # Angular errors must be absolute
    errs = [abs(x) for x in errs]

    try:
        return __get_ca_twoway(errs, lambda x: x[1:], lambda th: th >= thresh, filter_size)
    except StopIteration:
        raise ValueError('No angular errors => threshold')


def get_route_ridf(images, snap, step=1):
    return np.amin(get_ridf(images, snap, step), axis=0)


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


class Database:
    def __init__(self, path):
        self.entries = read_image_database(join('databases', path))

    def get_distance(self, i, j):
        '''Euclidean distance between two database entries (in m)'''
        dy = self.entries["y"][j] - self.entries["y"][i]
        dx = self.entries["x"][j] - self.entries["x"][i]
        return math.hypot(dy, dx)

    def get_distances(self, ref_entry, entries):
        dists = []
        for i in entries:
            dist = self.get_distance(ref_entry, i)
            dists.append(dist if i >= ref_entry else -dist)
        return dists

    def get_entry_bounds(self, max_dist, start_entry):
        '''Get upper and lower bounds for frames > max_dist from start frame'''
        upper_entry = start_entry
        while self.get_distance(start_entry, upper_entry) < max_dist:
            upper_entry += 1
        lower_entry = start_entry - 1
        while self.get_distance(start_entry, lower_entry) < max_dist:
            lower_entry -= 1
        return (lower_entry, upper_entry)

    def read_images(self, entries, preprocess=None):
        # Convert all the images to floats before we use them
        if preprocess is None:
            preprocess = improc.to_float
        else:
            preprocess = improc.chain(preprocess, improc.to_float)

        paths = self.entries["filepath"]
        if not isinstance(entries, Iterable):
            return read_images(paths[entries], preprocess)

        paths = [paths[entry] for entry in entries]
        return read_images(paths, preprocess)

    def plot_idfs(self, ax, ref_entry, max_dist, preprocess=None, fr_step=1, ridf_step=1):
        (lower, upper) = self.get_entry_bounds(max_dist, ref_entry)
        entries = range(lower, upper+fr_step, fr_step)
        dists = self.get_distances(ref_entry, entries)

        # Load snapshot and test images
        snap = self.read_images(ref_entry, preprocess)
        images = self.read_images(entries, preprocess)
        print("Testing frames %i to %i (n=%i)" %
              (lower, upper, images.shape[2]))

        # Show which part of route we're testing
        x = self.entries["x"]
        y = self.entries["y"]
        ax[1].plot(x, y, x[lower:upper], y[lower:upper],
                   x[ref_entry], y[ref_entry], 'ro')
        ax[1].set_xlabel("x (m)")
        ax[1].set_ylabel("y (m)")
        ax[1].axis("equal")

        # Plot (R)IDF diffs vs distance
        idf_diffs = get_route_idf(images, snap)
        ridf_diffs = get_route_ridf(images, snap, ridf_step)

        ax[0].plot(dists, idf_diffs, dists, ridf_diffs)
        ax[0].set_xlabel("Distance (m)")
        ax[0].set_xlim(-max_dist, max_dist)
        ax[0].set_xticks(range(-max_dist, max_dist))
        ax[0].set_ylabel("Mean image diff (px)")
        ax[0].set_ylim(0, 0.06)

    def get_test_frames(self, ref_entry, frame_dist, preprocess=None, fr_step=1):
        (lower, upper) = (ref_entry - frame_dist, ref_entry + frame_dist)
        entries = range(lower, upper+fr_step, fr_step)
        snap = self.read_images(ref_entry, preprocess)
        images = self.read_images(entries, preprocess)
        print("Testing frames %i to %i (n=%i)" %
              (lower, upper, images.shape[2]))
        return (images, snap, entries)

    def plot_idfs_frames(self, ref_entry, frame_dist, preprocess=None, fr_step=1, ridf_step=1):
        (images, snap, entries) = self.get_test_frames(
            ref_entry, frame_dist, preprocess, fr_step)

        idf_diffs = get_route_idf(images, snap)
        ridf_diffs = get_route_ridf(images, snap, ridf_step)
        plot_route_idf(entries, idf_diffs, ridf_diffs)
