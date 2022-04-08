import math

import bob_robotics.navigation as bobnav
import numpy as np

import navbench as nb


class Database(bobnav.Database):
    def calculate_distance(self, entry1, entry2):
        '''Euclidean distance between two database entries (in m)'''

        # CA bounds may be infinite so handle this
        if math.isinf(entry1) or math.isinf(entry2):
            return float('inf')

        return np.linalg.norm(np.array(self.position[entry1, 0:2]) - self.position[entry2, 0:2])

    def calculate_distances(self, ref_entry, entries):
        dists = []
        for i in entries:
            dist = self.calculate_distance(ref_entry, i)
            dists.append(dist if i >= ref_entry else -dist)
        return dists

    def calculate_heading_offset(self, distance_thresh):
        i = 0
        while self.distance[i] < distance_thresh:
            i += 1

        dpos = self.position[i, :] - self.position[0, :]
        return math.atan2(dpos[1], dpos[0])

    def entry_bounds(self, max_dist, start_entry):
        '''Get upper and lower bounds for frames > max_dist from start frame'''
        upper_entry = start_entry
        while self.calculate_distance(start_entry, upper_entry) < max_dist:
            upper_entry += 1
        lower_entry = start_entry
        while self.calculate_distance(start_entry, lower_entry) < max_dist:
            lower_entry -= 1
        return (lower_entry, upper_entry)

    def get_nearest_entries(self, x, y=None):
        if y is None:
            return self.entries.iloc[self.get_nearest_entries(x.x, x.y)]

        diff_x = np.atleast_2d(self.x) - np.atleast_2d(x).T
        diff_y = np.atleast_2d(self.y) - np.atleast_2d(y).T
        distances = np.hypot(diff_x, diff_y)
        nearest = np.argmin(distances, axis=1)
        assert len(nearest) == len(x)
        return nearest

    def load_test_frames(self, ref_entry, frame_dist, preprocess=None, fr_step=1):
        (lower, upper) = (ref_entry - frame_dist, ref_entry + frame_dist)
        entries = range(lower, upper+fr_step, fr_step)
        snap = self.read_images(ref_entry, preprocess)
        images = self.read_images(entries, preprocess)
        print("Testing frames %i to %i (n=%i)" %
              (lower, upper, len(images)))
        return (images, snap, entries)

    def plot_idfs_frames(self, ref_entry, frame_dist, preprocess=None, fr_step=1, ridf_step=1, filter_zeros=True):
        (images, snap, entries) = self.load_test_frames(
            ref_entry, frame_dist, preprocess, fr_step)

        idf_diffs = nb.route_idf(images, snap)
        ridf_diffs = nb.route_ridf(images, snap, ridf_step)
        nb.plot_route_idf(entries, idf_diffs, ridf_diffs,
                          filter_zeros=filter_zeros)

    def plot_idfs(self, ax, ref_entry, max_dist, preprocess=None, fr_step=1, ridf_step=1, filter_zeros=True):
        (lower, upper) = self.entry_bounds(max_dist, ref_entry)
        entries = range(lower, upper+fr_step, fr_step)
        dists = self.calculate_distances(ref_entry, entries)

        # Load snapshot and test images
        snap = self.read_images(ref_entry, preprocess)
        images = self.read_images(entries, preprocess)
        print("Testing frames %i to %i (n=%i)" %
              (lower, upper, len(images)))

        # Show which part of route we're testing
        ax[1].plot(self.x, self.y, self.position[lower:upper, 0], self.position[lower:upper, 1],
                   self.position[ref_entry, 0], self.position[ref_entry, 1], 'ro')
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
