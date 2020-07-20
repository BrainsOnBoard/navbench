from navbench import get_idf_ca
import pytest

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
