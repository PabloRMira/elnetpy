"Test linear elastic net"

import numpy as np
import pytest
from glmnet import ElasticNet

from elnetpy.linear import Elnet

SEED = 191


def test_elnet_lambdas():
    rng = np.random.default_rng(SEED)
    error = rng.normal(loc=0, scale=1, size=100)
    X = rng.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    lambdas = np.array([0, 0.5, 1, 2])
    elnet = Elnet(lambdas=lambdas)
    elnet.fit(X, y)


def test_elnet_one_lambda():
    rng = np.random.default_rng(SEED)
    error = rng.normal(loc=0, scale=1, size=100)
    X = rng.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    elnet = Elnet(lambdas=2)
    elnet.fit(X, y)


def test_elnet_n_lambda():
    rng = np.random.default_rng(SEED)
    error = rng.normal(loc=0, scale=1, size=100)
    X = rng.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    elnet = Elnet()
    elnet.fit(X, y)


@pytest.mark.parametrize("alpha", [0, 0.5, 1])
def test_elnet_glmnet(alpha):
    # compare to original glmnet Fortran algorithm
    rng = np.random.default_rng(SEED)
    error = rng.normal(loc=0, scale=1, size=100)
    X = rng.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    # glmnet
    m = ElasticNet(alpha=alpha)
    m.fit(X, y)
    # own implementation
    m2 = Elnet(alpha=alpha)
    m2.fit(X, y)
    # same lambda sequence
    np.testing.assert_almost_equal(m.lambda_path_, m2.lambda_path_)
    # same feature coefficients
    # decimal = 4: almost equal to the 4th decimal
    np.testing.assert_almost_equal(m.coef_path_, m2.coef_path_, decimal=4)
    # same intercept path
    np.testing.assert_almost_equal(m.intercept_path_, m2.intercept_path_, decimal=4)


@pytest.mark.parametrize("alpha", [-0.5, 0, 0.5, 1, 1.01])
def test_validate_elnet_alpha(alpha):
    if alpha < 0 or alpha > 1:
        with pytest.raises(ValueError):
            Elnet(alpha=alpha)
    else:
        Elnet(alpha=alpha)


@pytest.mark.parametrize("lambdas", [1, np.array([1, 2, 3]), "2"])
def test_validate_elnet_lambdas(lambdas):
    if isinstance(lambdas, str):
        with pytest.raises(ValueError):
            Elnet(lambdas=lambdas)
    else:
        Elnet(lambdas=lambdas)


@pytest.mark.parametrize("n_lambda", [100, 5, -1])
def test_validate_elnet_n_lambda(n_lambda):
    if isinstance(n_lambda, str) or n_lambda <= 5:
        with pytest.raises(ValueError):
            Elnet(n_lambda=n_lambda)
    else:
        Elnet(n_lambda=n_lambda)


@pytest.mark.parametrize("min_lambda_ratio", [1e-5, -1, 2, 0, 1])
def test_validate_elnet_min_lambda_ratio(min_lambda_ratio):
    if min_lambda_ratio <= 0 or min_lambda_ratio >= 1:
        with pytest.raises(ValueError):
            Elnet(min_lambda_ratio=min_lambda_ratio)
    else:
        Elnet(min_lambda_ratio=min_lambda_ratio)


@pytest.mark.parametrize("n_jobs", [-5, 0, 1, 2, 2.3, -1])
def test_validate_elnet_n_jobs(n_jobs):
    if n_jobs == 0 or n_jobs < -1 or not isinstance(n_jobs, int):
        with pytest.raises(ValueError):
            Elnet(n_jobs=n_jobs)
    else:
        Elnet(n_jobs=n_jobs)


@pytest.mark.parametrize("tol", [1e-5, 0, 1, 0.001])
def test_validate_elnet_tol(tol):
    if tol <= 0 or tol >= 1:
        with pytest.raises(ValueError):
            Elnet(tol=tol)
    else:
        Elnet(tol=tol)


@pytest.mark.parametrize("max_iter", [1, 100, 200, 50])
def test_validate_elnet_max_iter(max_iter):
    if max_iter < 100:
        with pytest.raises(ValueError):
            Elnet(max_iter=max_iter)
    else:
        Elnet(max_iter=max_iter)


@pytest.mark.parametrize("n_splits", [-1, 10, 5.5, 20, 500])
def test_validate_n_splits(n_splits):
    if n_splits <= 0 or n_splits >= 100 or not isinstance(n_splits, int):
        with pytest.raises(ValueError):
            Elnet(n_splits=n_splits)
    else:
        Elnet(n_splits=n_splits)


@pytest.mark.parametrize(
    "scoring", ["mean_squared_error", "mean_absolute_error", "something_else"]
)
def test_validate_scoring(scoring):
    if scoring == "something_else":
        with pytest.raises(ValueError):
            Elnet(scoring=scoring)
    else:
        Elnet(scoring=scoring)


@pytest.mark.parametrize("random_state", [1, 10, 7.5, 3])
def test_validate_random_state(random_state):
    if isinstance(random_state, float):
        with pytest.raises(ValueError):
            Elnet(random_state=random_state)
    else:
        Elnet(random_state=random_state)


def test_fit_cv():
    rng = np.random.default_rng(SEED)
    error = rng.normal(loc=0, scale=1, size=100)
    X = rng.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    m = Elnet(n_splits=3)
    m.fit(X, y)
    assert hasattr(m, "lambda_1se_")
    assert hasattr(m, "lambda_max_")
    assert hasattr(m, "cv_mean")
    assert hasattr(m, "cv_se")


def test_predict_default():
    rng = np.random.default_rng(SEED)
    error = rng.normal(loc=0, scale=1, size=100)
    X = rng.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    m = Elnet(n_splits=3)
    m.fit(X, y)
    m.predict(X)


def test_predict_custom_lambda():
    rng = np.random.default_rng(SEED)
    error = rng.normal(loc=0, scale=1, size=100)
    X = rng.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    m = Elnet(n_splits=3)
    m.fit(X, y)
    m.predict(X, lamb=1)


def test_predict_multiple_lambdas():
    rng = np.random.default_rng(SEED)
    error = rng.normal(loc=0, scale=1, size=100)
    X = rng.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    m = Elnet(n_splits=3)
    m.fit(X, y)
    preds = m.predict(X, lamb=[0.5, 1, 1.5])
    assert preds.shape == (100, 3)


def test_predict_without_cv():
    rng = np.random.default_rng(SEED)
    error = rng.normal(loc=0, scale=1, size=100)
    X = rng.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    m = Elnet(n_splits=1)
    m.fit(X, y)
    with pytest.raises(Exception):
        m.predict(X)


def test_fit_cv_glmnet_comparison():
    rng = np.random.default_rng(SEED)
    error = rng.normal(loc=0, scale=1, size=100)
    X = rng.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    m = Elnet(n_splits=3, random_state=182, scoring="r2")
    m.fit(X, y)
    m2 = ElasticNet(n_splits=3, random_state=182)
    m2.fit(X, y)
    np.testing.assert_almost_equal(m.lambda_max_, m2.lambda_max_)
    np.testing.assert_almost_equal(m.lambda_1se_, m2.lambda_best_[0])
