import numpy as np
from navbench import mean_absdiff

im0 = np.zeros((2, 2))
im1 = np.ones((2, 2))

arr0 = np.zeros((2, 2, 2))
arr1 = arr0.copy()
arr1[:, :, 1] = 1


def test_identity():
    assert mean_absdiff(im0, im0) == 0
    assert mean_absdiff(im1, im1) == 0


def test_other():
    assert mean_absdiff(im0, im1) == 1
    assert mean_absdiff(im1, im0) == 1


def test_scalar_vs_array():
    assert (mean_absdiff(im0, arr0) == [0, 0]).all()
    assert (mean_absdiff(arr0, im0) == [0, 0]).all()
    assert (mean_absdiff(im0, arr1) == [0, 1]).all()
    assert (mean_absdiff(arr1, im0) == [0, 1]).all()


def test_array_vs_array():
    assert (mean_absdiff(arr0, arr0) == [0, 0]).all()
    assert (mean_absdiff(arr0, arr1) == [0, 1]).all()
