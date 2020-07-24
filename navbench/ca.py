import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt


def __get_ca_bounds(vals, process_fun, thresh_fun, filter_size):
    '''
    Internal function. Get CA in leftward and rightward directions from goal
    position (taken as where the minimum value of vals is).
    '''

    # Convert to numpy array
    vals = np.array(vals)

    if vals.ndim != 1 or len(vals) == 0:
        raise ValueError('Input values must be vector')

    # Assume goal is where minimum is
    goal = np.argmin(vals)

    def filter_vals(vec):
        vec = process_fun(vec)

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

    # A StopIteration error is raised when there are no values for which
    # thresh_fun() == True. In this case the CA extends beyond the lower or
    # upper bounds of the input values and we indicate it by setting the bound
    # to None.
    try:
        lower = goal - get_ca(left)
    except StopIteration:
        lower = None
    try:
        upper = goal + get_ca(right)
    except StopIteration:
        upper = None

    return (lower, upper), goal


def __get_total_ca(bounds):
    assert len(bounds) == 2
    return bounds[1] - bounds[0]


def get_idf_ca_bounds(idf, filter_size=1):
    '''
    Get catchment area for 1D IDF.

    Differences from Andy's implementation:
        - I'm treating IDFs like this: [1, 2, 0, 2, 1] as having a CA of 2
          rather than 0.
        - Cases where vector length > filter size cause an error
    '''
    return __get_ca_bounds(idf, np.diff, lambda x: x < 0, filter_size)


def get_idf_ca(idf, filter_size=1):
    bounds, _ = get_idf_ca_bounds(idf, filter_size)
    return __get_total_ca(bounds)


def get_rca_bounds(errs, thresh=45, filter_size=1):
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
        return __get_ca_bounds(errs, lambda x: x[1:], lambda th: th >= thresh, filter_size)
    except StopIteration:
        raise ValueError('No angular errors => threshold')


def get_rca(errs, thresh=45, filter_size=1):
    bounds, _ = get_rca_bounds(errs, thresh, filter_size)
    return __get_total_ca(bounds)


def plot_ca(entries, vals, bounds, goal, filter_zeros=True, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # if filter_zeros:
    #     zero_idx = [i for i, val in enumerate(vals) if val == 0]
    #     vals[zero_idx] = None
    #     print('Warning: %i zero values (perfect matches?) are not being shown' %
    #           len(zero_idx))

    ax.plot(entries, vals)
    ax.plot(entries[bounds[0]:bounds[1]], vals[bounds[0]:bounds[1]], 'r')
    ax.set_xlim(entries[0], entries[-1])
    ax.plot([entries[goal], entries[goal]], ax.get_ylim(), 'k--')

    return ax
