import math
from os import listdir
from os.path import isfile, join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    df = df.rename(columns=lambda x: x.strip())  # strip whitespace from column headers

    entries = {
        "x": df["X [mm]"] / 1000,
        "y": df["Y [mm]"] / 1000,
        "z": df["Z [mm]"] / 1000,
        "heading": df["Heading [degrees]"],
        "filepath": [join(path, fn.strip()) for fn in df["Filename"]]
    }

    return entries


def read_images(paths, size=()):
    """Returns greyscale image(s) of type float."""

    # Single string as input
    if isinstance(paths, str):
        im = cv2.imread(paths, cv2.IMREAD_GRAYSCALE)

        # Convert to normalised matrix of floats
        info = np.iinfo(im.dtype)
        im = im.astype(np.float) / info.max

        return cv2.resize(im, size) if size else im

    if not paths:
        return []

    im0 = read_images(paths[0], size)
    images = np.zeros([im0.shape[0], im0.shape[1], len(paths)], dtype=np.float)
    images[:, :, 0] = im0
    for i in range(1, len(paths)):
        images[:, :, i] = read_images(paths[i], size)
    return images


def mean_absdiff(x, y):
    """Return mean absolute difference between two images or sets of images
    (as 3D matrices). """

    # Check that the images are of the same size
    assert x.shape[0:2] == y.shape[0:2]

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


def get_route_ridf(images, snap, step=1):
    return np.amin(get_ridf(images, snap, step), axis=0)


class Database:
    def __init__(self, path, size=(), step=1):
        self.entries = read_image_database(join('databases', path))
        self.size = size
        self.step = step

    def get_distance(self, i, j):
        '''Euclidean distance between two database entries (in m)'''
        dy = self.entries["y"][j] - self.entries["y"][i]
        dx = self.entries["x"][j] - self.entries["x"][i]
        return math.hypot(dy, dx)

    def get_distances(self, from_entry, to_entry, ref_entry):
        assert to_entry >= from_entry

        dists = []
        for i in range(from_entry, to_entry, self.step):
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

    def read_images(self, from_entry, to_entry=None):
        paths = self.entries["filepath"]
        if to_entry is None:
            return read_images(paths[from_entry], self.size)

        return read_images(paths[from_entry:to_entry:self.step], self.size)

    def plot_idfs(self, ax, ref_entry, max_dist, ridf_step=1):
        (lower, upper) = self.get_entry_bounds(max_dist, ref_entry)
        dists = self.get_distances(lower, upper, ref_entry)

        # Load snapshot and test images
        snap = self.read_images(ref_entry)
        images = self.read_images(lower, upper)
        print("Testing frames %i to %i (n=%i)" % (lower, upper, images.shape[2]))

        # Show which part of route we're testing
        x = self.entries["x"]
        y = self.entries["y"]
        ax[1].plot(x, y, x[lower:upper], y[lower:upper], x[ref_entry], y[ref_entry], 'ro')
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

    def plot_idfs_frames(self, ref_entry, frame_dist, ridf_step=1):
        (lower, upper) = (ref_entry - frame_dist, ref_entry + frame_dist)
        snap = self.read_images(ref_entry)
        images = self.read_images(lower, upper)
        print("Testing frames %i to %i (n=%i)" % (lower, upper, images.shape[2]))

        idf_diffs = get_route_idf(images, snap)
        ridf_diffs = get_route_ridf(images, snap, ridf_step)
        fr_rng = range(lower, upper, self.step)
        plt.plot(fr_rng, idf_diffs, fr_rng, ridf_diffs)
        plt.xlabel("Frame")
        plt.xlim(lower, upper)
        plt.ylabel("Mean image diff (px)")
        plt.ylim(0, plt.ylim()[1])
