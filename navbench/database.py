import math
import os
from collections.abc import Iterable

import navbench as nb
from navbench import improc

class Database:
    def __init__(self, path, fullpath=False):
        if not fullpath:
            path = os.path.join('databases', path)
        self.entries = nb.read_image_database(path)

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
            return nb.read_images(paths[entries], preprocess)

        paths = [paths[entry] for entry in entries]
        return nb.read_images(paths, preprocess)

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
        idf_diffs = nb.get_route_idf(images, snap)
        ridf_diffs = nb.get_route_ridf(images, snap, ridf_step)

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

        idf_diffs = nb.get_route_idf(images, snap)
        ridf_diffs = nb.get_route_ridf(images, snap, ridf_step)
        nb.plot_route_idf(entries, idf_diffs, ridf_diffs)
