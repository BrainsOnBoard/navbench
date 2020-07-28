import math
from os import listdir
from os.path import isfile, join
import pathlib
from collections.abc import Iterable

import cv2
import numpy as np
import pandas as pd

import navbench as nb
from navbench import improc


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
        assert im is not None  # Check im loaded successfully

        # Run preprocessing step on images
        if preprocess:
            im = preprocess(im)

        return im

    return [read_images(path, preprocess) for path in paths]


class Database:
    def __init__(self, path, fullpath=False):
        if not fullpath:
            MODPATH = pathlib.Path(__file__).resolve().parent
            path = join(MODPATH.parent, 'databases', path)
        self.entries = nb.read_image_database(path)

    def __len__(self):
        return len(self.entries["filepath"])

    def distance(self, i, j):
        '''Euclidean distance between two database entries (in m)'''
        dy = self.entries["y"][j] - self.entries["y"][i]
        dx = self.entries["x"][j] - self.entries["x"][i]
        return math.hypot(dy, dx)

    def distances(self, ref_entry, entries):
        dists = []
        for i in entries:
            dist = self.distance(ref_entry, i)
            dists.append(dist if i >= ref_entry else -dist)
        return dists

    def entry_bounds(self, max_dist, start_entry):
        '''Get upper and lower bounds for frames > max_dist from start frame'''
        upper_entry = start_entry
        while self.distance(start_entry, upper_entry) < max_dist:
            upper_entry += 1
        lower_entry = start_entry - 1
        while self.distance(start_entry, lower_entry) < max_dist:
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
            return nb.read_images(paths[entries], preprocess)

        paths = [paths[entry] for entry in entries]
        return nb.read_images(paths, preprocess)

    def plot_idfs(self, ax, ref_entry, max_dist, preprocess=None, fr_step=1, ridf_step=1, filter_zeros=True):
        (lower, upper) = self.entry_bounds(max_dist, ref_entry)
        entries = range(lower, upper+fr_step, fr_step)
        dists = self.distances(ref_entry, entries)

        # Load snapshot and test images
        snap = self.read_images(ref_entry, preprocess)
        images = self.read_images(entries, preprocess)
        print("Testing frames %i to %i (n=%i)" %
              (lower, upper, len(images)))

        # Show which part of route we're testing
        x = self.entries["x"]
        y = self.entries["y"]
        ax[1].plot(x, y, x[lower:upper], y[lower:upper],
                   x[ref_entry], y[ref_entry], 'ro')
        ax[1].set_xlabel("x (m)")
        ax[1].set_ylabel("y (m)")
        ax[1].axis("equal")

        # Plot (R)IDF diffs vs distance
        idf_diffs = nb.route_idf(images, snap)
        ridf_diffs = nb.route_ridf(images, snap, ridf_step)

        if filter_zeros:
            idf_diffs = nb.zeros_to_nones(idf_diffs)
            ridf_diffs = nb.zeros_to_nones(ridf_diffs)
        ax[0].plot(dists, idf_diffs, dists, ridf_diffs)
        ax[0].set_xlabel("Distance (m)")
        ax[0].set_xlim(-max_dist, max_dist)
        ax[0].set_xticks(range(-max_dist, max_dist))
        ax[0].set_ylabel("Mean image diff (px)")
        ax[0].set_ylim(0, 0.06)

    def test_frames(self, ref_entry, frame_dist, preprocess=None, fr_step=1):
        (lower, upper) = (ref_entry - frame_dist, ref_entry + frame_dist)
        entries = range(lower, upper+fr_step, fr_step)
        snap = self.read_images(ref_entry, preprocess)
        images = self.read_images(entries, preprocess)
        print("Testing frames %i to %i (n=%i)" %
              (lower, upper, len(images)))
        return (images, snap, entries)

    def plot_idfs_frames(self, ref_entry, frame_dist, preprocess=None, fr_step=1, ridf_step=1, filter_zeros=True):
        (images, snap, entries) = self.test_frames(
            ref_entry, frame_dist, preprocess, fr_step)

        idf_diffs = nb.route_idf(images, snap)
        ridf_diffs = nb.route_ridf(images, snap, ridf_step)
        nb.plot_route_idf(entries, idf_diffs, ridf_diffs, filter_zeros=filter_zeros)
