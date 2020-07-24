from navbench import ridf
import numpy as np

im0 = np.zeros((2, 2))

def test_autoridf():
    assert (ridf(im0, im0) == [0, 0]).all()
