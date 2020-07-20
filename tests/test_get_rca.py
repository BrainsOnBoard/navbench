from navbench import get_rca
import pytest

def test_empty():
    assert get_rca([]) == 0

def test_bad_thresh():
    with pytest.raises(AssertionError):
        assert get_rca([], -1)

def test_typical():
    assert get_rca([45, 10, 0, 10, 45]) == 2

def test_negative_angles():
    assert get_rca([-45, 10, 0, 10, 45]) == 2
    assert get_rca([45, 10, 0, 10, -45]) == 2

def test_medfilt_right():
    assert get_rca([0, 10, 45, 45], filter_size=3) == 1

def test_medfilt_left():
    assert get_rca([45, 45, 10, 0], filter_size=3) == 1

def test_medfilt_both():
    assert get_rca([45, 45, 10, 0, 10, 45, 45], filter_size=3) == 2
