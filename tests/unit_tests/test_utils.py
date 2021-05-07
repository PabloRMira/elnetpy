"Test helper functions"

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, mean_absolute_error

from elnetpy.utils import (
    standardize_inputs,
    destandardize_coefs,
    get_linear_scorer,
    score_multiple,
    check_fitted_model,
    interpolate_model,
)


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


def test_get_linear_scorer():
    mae = get_linear_scorer["mean_absolute_error"]
    assert mae["scorer"] == mean_absolute_error
    assert not mae["greater_better"]


def test_score_multiple():
    y_true = np.array([1, 5, 2, 3])
    y_preds_mat = np.array([[1, 8, 2], [8, 21, 1], [8, 1, 5], [10, 3, 1]])
    scores = score_multiple(y_true, y_preds_mat, mean_squared_error)
    expected_scores = (
        ((y_preds_mat - y_true.reshape((4, 1))) ** 2).mean(axis=0).tolist()
    )
    assert scores == expected_scores


def test_check_fitted_model():
    unfitted_model = type("unfitted_model", (object,), {"something": 1})
    fitted_model = type("unfitted_model", (object,), {"lambda_path_": [1, 2, 3]})
    check_fitted_model(fitted_model)
    with pytest.raises(Exception):
        check_fitted_model(unfitted_model)


def test_interpolate_model():
    lamb = 0.5
    lambda_path = np.array([0, 1])
    coef_path = np.array([[10, 20], [20, 40]])
    intercept_path = np.array([5, 10])
    coef, intercept = interpolate_model(lamb, lambda_path, coef_path, intercept_path)
    expected_coef = np.array([15, 30])
    expected_intercept = np.array(7.5)
    np.testing.assert_equal(coef, expected_coef)
    np.testing.assert_equal(intercept, expected_intercept)
