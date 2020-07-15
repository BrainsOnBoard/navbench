import math
import matplotlib.pyplot as plt
import navbench
import os

class Database:
    def __init__(self, path, size=(), step=1):
        self.entries = navbench.read_image_database(os.path.join('databases', path))
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
            return navbench.read_image(paths[from_entry], self.size)

        return navbench.read_images(paths[from_entry:to_entry:self.step], self.size)
