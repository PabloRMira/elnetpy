"Performance test against glmnet"

import itertools
import numpy as np
import pytest
from glmnet import ElasticNet

from elnetpy.linear import Elnet

SEED = 191


@pytest.mark.parametrize(
    "alpha, n_obs", itertools.product([0, 0.5, 1], [100, 1000, 10000])
)
def test_linear_glmnet(benchmark, alpha, n_obs):
    rng = np.random.default_rng(SEED)
    error = rng.normal(loc=0, scale=1, size=n_obs)
    X = rng.normal(loc=5, scale=2, size=(n_obs, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    m = ElasticNet(alpha=alpha)
    benchmark(m.fit, X, y)


@pytest.mark.parametrize(
    "alpha, n_obs", itertools.product([0, 0.5, 1], [100, 1000, 10000])
)
def test_linear_elnetpy(benchmark, alpha, n_obs):
    rng = np.random.default_rng(SEED)
    error = rng.normal(loc=0, scale=1, size=n_obs)
    X = rng.normal(loc=5, scale=2, size=(n_obs, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    m2 = Elnet(alpha=alpha)
    benchmark(m2.fit, X, y)
