import numpy as np
import elnetpy


def test_matrix_mult():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a = np.array([[1], [2], [3]])
    np_dot = np.dot(x, a)
    el_dot = elnetpy.matrix_mult(x, a)
    assert np_dot == el_dot
