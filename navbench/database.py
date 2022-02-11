import math
import os
from collections.abc import Iterable

import cv2
import numpy as np
import pandas as pd
import yaml

import navbench as nb
from navbench import imgproc


def clean_entries(val):
    val = val.strip()
    if val == "":
        return float('nan')
    else:
        return float(val)

def read_image_database(path, limits_metres):
    """Read info for image database entries from CSV file."""

    print('Loading database at %s...' % path)

    try:
        df = pd.read_csv(os.path.join(path, "database_entries.csv"))
    except FileNotFoundError:
        print("Warning: No CSV file found for", path)
        assert limits_metres is None

        fnames = [f for f in os.listdir(path) if f.endswith(
            '.png') or f.endswith('.jpg')]

        # If there's no CSV file then just treat all the files in the folder as
        # separate entries. Note that the files will be in alphabetical order,
        # which may not be what you want!
        entries = {
            "filepath": sorted([os.path.join(path, f) for f in fnames])
        }

        return entries

    # strip whitespace from column headers
    df = df.rename(columns=lambda x: x.strip())

    # strip whitespace from entries and convert empty strings to NaNs
    for col in ("X [mm]", "Y [mm]", "Z [mm]"):
        if df[col].dtype != float:
            df[col] = df[col].apply(clean_entries)

    position = np.array([df["X [mm]"], df["Y [mm]"], df["Z [mm]"]], dtype=float).transpose()
    position /= 1000  # Convert to m

    # Calculate cumulative distance for each point on route
    elem_dists = np.hypot(np.diff(position[:,0]), np.diff(position[:,1]))
    distance = np.nancumsum([0, *elem_dists])

    # User has specified limits
    if limits_metres is not None:
        assert len(limits_metres) == 2
        assert limits_metres[0] >= 0
        assert limits_metres[0] < limits_metres[1]
        if limits_metres[1] > distance[-1]:
            print(f'Warning: limit of {limits_metres[1]} is greater than route length of {distance[1]}')

        sel = np.logical_and(distance >= limits_metres[0], distance < limits_metres[1])
        position = position[sel]
        distance = distance[sel]
        df = df[sel]

    entries = {
        "position": position,
        "heading": df["Heading [degrees]"],
        "distance": distance
    }

    # Note that there won't be a Filename column in video-type databases
    if 'Filename' in df:
        entries["filepath"] = [os.path.join(path, fn.strip()) for fn in df["Filename"]]
    else:
        entries["filepath"] = [None] * len(df)

    entries["position"] /= 1000  # Convert to m

    print('Database contains %d images' % len(entries['filepath']))

    return entries


def apply_functions(im, funs):
    if funs is None:
        return im
    if isinstance(funs, Iterable):
        for fun in funs:
            im = apply_functions(im, fun)
        return im
    return funs(im)


def read_images(paths, preprocess=None, greyscale=True):
    # Single string as input
    if isinstance(paths, str):
        if greyscale:
            flags = cv2.IMREAD_GRAYSCALE
        else:
            flags = cv2.IMREAD_COLOR
        im = cv2.imread(paths, flags)
        assert im is not None  # Check im loaded successfully

        # Run preprocessing step on images
        im = apply_functions(im, preprocess)

        return im

    # Otherwise, we have a collection of paths
    return [read_images(path, preprocess, greyscale) for path in paths]


class Database:
    def __init__(self, path, limits_metres=None):
        self.path = path
        self.name = os.path.basename(path)

        # Turn the elements of the dict into object attributes
        entries = nb.read_image_database(path, limits_metres)
        for key, value in entries.items():
            setattr(self, key, value)

        if hasattr(self, "position"):
            # Add these attributes for convenience
            self.x = self.position[:, 0]
            self.y = self.position[:, 1]
            self.z = self.position[:, 2]

        metadata_path = os.path.join(path, "database_metadata.yaml")
        try:
            with open(metadata_path, 'r') as file:
                # OpenCV puts crap on the first two lines of the file; skip them
                file.readline()
                file.readline()
                self.metadata = yaml.full_load(file)["metadata"]
        except:
            self.metadata = None
            print("WARNING: Could not read database_metadata.yaml")

        if self.metadata and 'needsUnwrapping' in self.metadata and self.metadata['needsUnwrapping']:
            print("!!!!! WARNING: This database has not been unwrapped. " +
                  "Analysis may not make sense! !!!!!")

    def __len__(self):
        return len(self.filepath)

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

    def read_images(self, entries=None, preprocess=None, to_float=True, greyscale=True):
        if not greyscale:
            # For colour images, it makes more sense to use an RGB colour space in python
            preprocess = (preprocess, lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        if to_float:
            # Convert all the images to floats before we use them
            preprocess = (preprocess, imgproc.to_float)

        paths = self.filepath
        if not entries is None:  # (otherwise load all images)
            if not isinstance(entries, Iterable):
                return nb.read_images(paths[entries], preprocess, greyscale)

            paths = [paths[entry] for entry in entries]
        return nb.read_images(paths, preprocess, greyscale)
