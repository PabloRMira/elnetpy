"Test linear elastic net"

import numpy as np
import pytest
from glmnet import ElasticNet

from elnetpy.linear import Elnet


def test_elnet_lambdas():
    error = np.random.normal(loc=0, scale=1, size=100)
    X = np.random.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    lambdas = np.array([0, 0.5, 1, 2])
    elnet = Elnet(lambdas=lambdas)
    elnet.fit(X, y)


def test_elnet_one_lambda():
    error = np.random.normal(loc=0, scale=1, size=100)
    X = np.random.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    elnet = Elnet(lambdas=2)
    elnet.fit(X, y)


def test_elnet_n_lambda():
    error = np.random.normal(loc=0, scale=1, size=100)
    X = np.random.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    elnet = Elnet()
    elnet.fit(X, y)


@pytest.mark.parametrize("alpha", [0, 0.5, 1])
def test_elnet_glmnet(alpha):
    # compare to original glmnet Fortran algorithm
    error = np.random.normal(loc=0, scale=1, size=100)
    X = np.random.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    # glmnet
    m = ElasticNet(alpha=alpha)
    m.fit(X, y)
    # own implementation
    m2 = Elnet(alpha=alpha)
    m2.fit(X, y)
    # glmnet has a stopping criterion we have not implemented yet
    n_lambda_glmnet = len(m.lambda_path_)
    # same lambda sequence
    np.testing.assert_almost_equal(m.lambda_path_, m2.lambda_path_[0:n_lambda_glmnet])
    # same feature coefficients
    # decimal = 4: almost equal to the 4th decimal
    np.testing.assert_almost_equal(
        m.coef_path_, m2.coef_path_[:, 0:n_lambda_glmnet], decimal=4
    )
    # same intercept path
    np.testing.assert_almost_equal(
        m.intercept_path_, m2.intercept_path_[0:n_lambda_glmnet], decimal=4
    )
