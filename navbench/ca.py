import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

import navbench as nb


class CatchmentArea:
    def __init__(self, vals, process_fun, thresh_fun, goal_idx, medfilt_size):
        # Convert to numpy array
        vals = np.array(vals)

        if vals.ndim != 1 or len(vals) == 0:
            raise ValueError('Input values must be vector')

        if goal_idx is None:
            # Assume goal is where minimum is
            goal_idx = np.argmin(vals)

        # Do median filtering
        self.filtered_vals = medfilt(vals, medfilt_size)

        # Apply process_fun to values from left and right of goal
        left = self.filtered_vals[goal_idx::-1]
        if len(left) > 0:
            left = process_fun(left)
        right = self.filtered_vals[goal_idx:]
        if len(right) > 0:
            right = process_fun(right)

        def ca(vec):
            if not vec.size:  # Empty array
                return 0

            # Return index of first value in vec for which thresh_fun() returns true
            return next(i for i, val in enumerate(vec) if thresh_fun(val))

        # A StopIteration error is raised when there are no values for which
        # thresh_fun() == True. In this case the CA extends beyond the lower or
        # upper bounds of the input values and we indicate it by setting the bound
        # to None.
        try:
            lower = goal_idx - ca(left)
        except StopIteration:
            lower = None
        try:
            upper = goal_idx + ca(right)
        except StopIteration:
            upper = None

        self.bounds = (lower, upper)
        self.goal_idx = goal_idx
        self.vals = vals

    def size(self):
        return self.bounds[1] - self.bounds[0]

    def plot(self, entries, filter_zeros=True, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        if filter_zeros:
            zeros = self.vals == 0
            self.vals = np.array(self.vals)
            self.vals[zeros] = None
            self.filtered_vals[zeros] = None
            print(sum(zeros), 'zero values are not being shown')

        lines = ax.plot(entries, self.vals)
        if self.filtered_vals is not None:
            ax.plot(entries, self.filtered_vals,
                    ':', color=lines[0].get_color())
        ax.plot(entries[self.bounds[0]:self.bounds[1]],
                self.vals[self.bounds[0]:self.bounds[1]], 'r')
        ax.set_xlim(entries[0], entries[-1])
        ax.set_ylim(bottom=0)
        ax.plot([entries[self.goal_idx], entries[self.goal_idx]],
                ax.get_ylim(), 'k--')

        if filter_zeros:
            for entry, val in zip(entries, self.vals):
                if val is None:
                    plt.plot((entry, entry), ax.get_ylim(), 'r:')

        return ax


def calculate_ca(idf, goal_idx=None, medfilt_size=1):
    '''
    Get catchment area for 1D IDF.

    Differences from Andy's implementation:
        - I'm treating IDFs like this: [1, 2, 0, 2, 1] as having a CA of 2
          rather than 0.
        - Cases where vector length > filter size cause an error
    '''
    return CatchmentArea(idf, np.diff, lambda x: x < 0, goal_idx, medfilt_size)


def calculate_rca(errs, thresh=45, goal_idx=None, medfilt_size=1):
    '''
    Get rotational catchment area:
        i.e., area over which errs < some_threshold

    Differences from Andy's implementation:
        - medfilt_size defaults to 1, not 3
    '''
    assert thresh >= 0

    return CatchmentArea(errs, lambda x: x[1:], lambda th: th >= thresh,
                         goal_idx, medfilt_size)
