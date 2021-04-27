from navbench import ridf
import numpy as np

im0 = np.zeros((4, 2))
im1 = np.zeros((4, 3))
im2 = im1.copy()
im1[:, 0] = 1
im2[:, 1] = 1
im3 = np.zeros((5, 4))
im3[:, 2] = 1

def compare_mat(a, b):
    a = np.array(a)
    b = np.array(b)
    assert a.ndim == b.ndim
    assert a.shape == b.shape
    assert (a == b).all()


def test_autoridf():
    compare_mat(ridf(im0, im0), [0, 0])
    compare_mat(ridf(im1, im1), [0, 2/3, 2/3])


def test_other():
    compare_mat(ridf(im1, im2), [2/3, 0, 2/3])
    compare_mat(ridf(im2, im1), [2/3, 2/3, 0])


def test_multi():
    compare_mat(ridf([im1, im2], im1), [[0, 2/3, 2/3], [2/3, 2/3, 0]])

    res = ridf([im1, im1, im2], im1)
    compare_mat(res, [[0, 2/3, 2/3], [0, 2/3, 2/3], [2/3, 2/3, 0]])


def test_step2():
    compare_mat(ridf(im0, im0, step=2), [0])
    compare_mat(ridf(im3, im3, step=2), [0, 1/2])
