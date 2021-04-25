"Test helper functions"

import numpy as np

from elnetpy.utils import standardize_inputs, destandardize_coefs


def test_standardize_inputs():
    X = np.array(
        [
            [1, 2, 3],
            [-1, 5, 10],
            [-10, 2, 8],
        ]
    )
    y = np.array([10, 32, 12])
    expected_X = np.zeros([3, 3])
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col - np.mean(col)
        col = col / np.sqrt(np.mean(col ** 2))
        expected_X[:, j] = col
    expected_y = y - np.mean(y)
    y_std = np.sqrt(np.mean(expected_y ** 2))
    expected_y = expected_y / y_std
    Xt, yt = standardize_inputs(X, y)
    np.testing.assert_equal(Xt, expected_X)
    np.testing.assert_equal(yt, expected_y)


def test_standardize_inputs_stds():
    X = np.array(
        [
            [1, 2, 3],
            [-1, 5, 10],
            [-10, 2, 8],
        ]
    )
    y = np.array([10, 32, 12])
    expected_X = np.zeros([3, 3])
    expected_X_stds = np.zeros(3)
    expected_X_means = np.zeros(3)
    for j in range(X.shape[1]):
        col = X[:, j]
        expected_X_means[j] = np.mean(col)
        col = col - expected_X_means[j]
        expected_X_stds[j] = np.sqrt(np.mean(col ** 2))
        col = col / expected_X_stds[j]
        expected_X[:, j] = col
    expected_y_mean = np.mean(y)
    expected_y = y - expected_y_mean
    expected_y_std = np.sqrt(np.mean(expected_y ** 2))
    expected_y = expected_y / expected_y_std
    Xt, yt, X_means, X_stds, y_mean, y_std = standardize_inputs(
        X, y, return_means_stds=True
    )
    np.testing.assert_equal(Xt, expected_X)
    np.testing.assert_equal(yt, expected_y)
    np.testing.assert_equal(X_means, expected_X_means)
    np.testing.assert_equal(X_stds, expected_X_stds)
    assert y_mean == expected_y_mean
    assert y_std == expected_y_std


def test_destandardize_coefs():
    # simulated standardized coefficients for 5 features and 100 lambdas
    coefs_mat = np.random.normal(size=(5, 100))
    X_means = np.array([3, 2, 5, 1, 0])
    X_stds = np.array([2, 0.5, 1, 5, 0.2])
    y_mean = 12
    y_std = 0.4
    expected_coefs_mat_destd = coefs_mat.copy()
    expected_intercepts = np.zeros(100)
    for k in range(coefs_mat.shape[1]):
        expected_intercepts[k] = y_mean
        for j in range(coefs_mat.shape[0]):
            expected_coefs_mat_destd[j, k] = (
                expected_coefs_mat_destd[j, k] * y_std
            ) / X_stds[j]
            expected_intercepts[k] -= expected_coefs_mat_destd[j, k] * X_means[j]
    coefs_mat_destd, intercepts = destandardize_coefs(
        coefs_mat, X_means, X_stds, y_mean, y_std
    )
    np.testing.assert_equal(coefs_mat_destd, expected_coefs_mat_destd)
    np.testing.assert_almost_equal(intercepts, expected_intercepts)
