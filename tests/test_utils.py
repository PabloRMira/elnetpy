"Test helper functions"

import numpy as np

from elnetpy.utils import standardize_inputs


def test_standardize_inputs():
    X = np.array(
        [
            [1, 2, 3],
            [-1, 5, 10],
            [-10, 2, 8],
        ]
    )
    y = np.array([10, 32, 12])
    lambdas = np.array([0, 0.5, 1])
    expected_X = np.zeros([3, 3])
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col - np.mean(col)
        col = col / np.sqrt(np.mean(col ** 2))
        expected_X[:, j] = col
    expected_y = y - np.mean(y)
    y_std = np.sqrt(np.mean(expected_y ** 2))
    expected_y = expected_y / y_std
    expected_lambdas = lambdas / y_std
    Xt, yt, lambdast = standardize_inputs(X, y, lambdas)
    np.testing.assert_equal(Xt, expected_X)
    np.testing.assert_equal(yt, expected_y)
    np.testing.assert_equal(lambdast, expected_lambdas)
