from navbench import ridf
import numpy as np

im0 = np.zeros((4, 2))
im1 = np.zeros((4, 3))
im2 = im1.copy()
im1[:, 0] = 1
im2[:, 1] = 1
im3 = np.zeros((5, 4))
im3[:, 2] = 1


def test_autoridf():
    assert (ridf(im0, im0) == [0, 0]).all()
    assert (ridf(im1, im1) == [0, 2/3, 2/3]).all()


def test_other():
    assert (ridf(im1, im2) == [2/3, 0, 2/3]).all()
    assert (ridf(im2, im1) == [2/3, 2/3, 0]).all()


def test_multi():
    assert (ridf([im1, im2], im1) == [[0, 2/3, 2/3], [2/3, 2/3, 0]]).all()

    res = ridf([im1, im1, im2], im1)
    assert res.shape == (3, 3)
    assert (res == [[0, 2/3, 2/3], [0, 2/3, 2/3], [2/3, 2/3, 0]]).all()


def test_step2():
    assert (ridf(im0, im0, 2) == [0]).all()
    assert (ridf(im3, im3, 2) == [0, 1/2]).all()
