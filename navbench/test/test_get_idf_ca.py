import pytest
from navbench.ca import *


def test_typical():
    idf = [1, 2, 0, 2, 3, 1]
    bounds, goal_idx, *_ = idf_ca_bounds(idf)
    assert goal_idx == 2
    assert bounds == (1, 4)
    assert idf_ca(idf) == 3

def test_explicit_goal():
    idf = [1, 2, 0, 2, 3, 1]
    bounds, *_ = idf_ca_bounds(idf, 2)
    assert bounds == (1, 4)


def test_no_left():
    idf = [0, 2, 1]
    bounds, goal_idx, *_ = idf_ca_bounds(idf)
    assert goal_idx == 0
    assert bounds == (0, 1)
    assert idf_ca(idf) == 1


def test_no_right():
    idf = [1, 2, 0]
    bounds, goal_idx, *_ = idf_ca_bounds(idf)
    assert goal_idx == 2
    assert bounds == (1, 2)
    assert idf_ca(idf) == 1


def test_empty():
    with pytest.raises(ValueError):
        idf_ca([])


def test_single():
    assert idf_ca([0]) == 0


def test_infinite_left():
    idf = [1, 1, 0, 2, 1]
    bounds, goal_idx, *_ = idf_ca_bounds(idf)
    assert goal_idx == 2
    assert bounds == (None, 3)


def test_infinite_right():
    idf = [1, 2, 0, 1, 1]
    bounds, goal_idx, *_ = idf_ca_bounds(idf)
    assert goal_idx == 2
    assert bounds == (1, None)


def test_medfilt_right():
    assert idf_ca([0, 1, 2, 1, 3, 1], medfilt_size=3) == 3


def test_medfilt_left():
    assert idf_ca([1, 3, 1, 2, 1, 0], medfilt_size=3) == 3


def test_medfilt_both():
    assert idf_ca([1, 3, 1, 2, 1, 0, 1, 2, 1, 3, 1], medfilt_size=3) == 6
