import pytest
from navbench import get_idf_ca

def test_typical():
    assert get_idf_ca([1, 2, 0, 2, 3, 1]) == 3

def test_no_left():
    assert get_idf_ca([0, 2, 1]) == 1

def test_no_right():
    assert get_idf_ca([1, 2, 0]) == 1

def test_empty():
    assert get_idf_ca([]) == 0

def test_single():
    assert get_idf_ca([0]) == 0

def test_infinite_left():
    with pytest.raises(ValueError):
        get_idf_ca([1, 1, 0, 2, 1])

def test_infinite_right():
    with pytest.raises(ValueError):
        get_idf_ca([1, 2, 0, 1, 1])

def test_medfilt_toosmall():
    with pytest.raises(ValueError):
        get_idf_ca([1, 2, 0, 2, 1], filter_size=3)

def test_medfilt_right():
    assert get_idf_ca([0, 1, 2, 1, 3, 1], filter_size=3) == 3

def test_medfilt_left():
    assert get_idf_ca([1, 3, 1, 2, 1, 0], filter_size=3) == 3

def test_medfilt_both():
    assert get_idf_ca([1, 3, 1, 2, 1, 0, 1, 2, 1, 3, 1], filter_size=3) == 6

def test_medfilt_empty():
    assert get_idf_ca([], filter_size=3) == 0
