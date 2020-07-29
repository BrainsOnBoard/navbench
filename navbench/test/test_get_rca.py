import pytest
from navbench.ca import *


def test_bad_thresh():
    with pytest.raises(AssertionError):
        assert rca([0, 0, 0], -1)


def test_typical():
    errs = [45, 10, 0, 10, 45]
    bounds, goal_idx, *_ = rca_bounds(errs)
    assert goal_idx == 2
    assert bounds == (1, 3)
    assert rca(errs) == 2


def test_explicit_goal():
    errs = [45, 10, 0, 10, 45]
    bounds2, *_ = rca_bounds(errs, 45, 2)
    assert bounds2 == (1, 3)


def test_medfilt_right():
    bounds, goal_idx, *_ = rca_bounds([0, 46, 30, 30, 45, 45], medfilt_size=3)
    assert goal_idx == 0
    assert bounds == (0, 3)


def test_medfilt_left():
    bounds, goal_idx, *_ = rca_bounds([45, 45, 30, 30, 46, 0], medfilt_size=3)
    assert goal_idx == 5
    assert bounds == (2, 5)


def test_medfilt_both():
    errs = [45, 45, 30, 30, 46, 0, 46, 30, 30, 45, 45]
    bounds, goal_idx, *_ = rca_bounds(errs, medfilt_size=3)
    assert goal_idx == 5
    assert bounds == (2, 8)
