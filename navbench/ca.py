import numpy as np
from scipy.signal import medfilt

def __get_ca_twoway(vals, filter_fun, thresh_fun, filter_size):
    '''
    Internal function. Get total CA in leftward and rightward direction from
    goal position (taken as where the minimum value of vals is).
    '''

    # Must be numpy array
    vals = np.array(vals)

    # If vals is empty
    if not vals.size:
        return 0

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
