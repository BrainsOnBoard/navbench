import pytest
from navbench.ca import *


def test_bad_thresh():
    with pytest.raises(AssertionError):
        assert get_rca([0, 0, 0], -1)


def test_typical():
    errs = [45, 10, 0, 10, 45]
    bounds, goal = get_rca_bounds(errs)
    assert goal == 2
    assert bounds == (1, 3)
    assert get_rca(errs) == 2


def test_negative_angles():
    errs1 = [-45, 10, 0, 10, 45]
    bounds1, goal1 = get_rca_bounds(errs1)
    assert goal1 == 2
    assert bounds1 == (1, 3)

    errs2 = [45, 10, 0, 10, -45]
    bounds2, goal2 = get_rca_bounds(errs2)
    assert goal2 == 2
    assert bounds2 == (1, 3)


def test_medfilt_right():
    bounds, goal = get_rca_bounds([0, 46, 30, 30, 45, 45], filter_size=3)
    assert goal == 0
    assert bounds == (0, 3)


def test_medfilt_left():
    bounds, goal = get_rca_bounds([45, 45, 30, 30, 46, 0], filter_size=3)
    assert goal == 5
    assert bounds == (2, 5)


def test_medfilt_both():
    errs = [45, 45, 30, 30, 46, 0, 46, 30, 30, 45, 45]
    bounds, goal = get_rca_bounds(errs, filter_size=3)
    assert goal == 5
    assert bounds == (2, 8)
