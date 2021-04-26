"Performance test against glmnet"

import numpy as np
import pytest
from glmnet import ElasticNet

from elnetpy.linear import Elnet

error = np.random.normal(loc=0, scale=1, size=100)
X = np.random.normal(loc=5, scale=2, size=(100, 4))
true_betas = np.array([1, -2, 0.5, 1])
y = X.dot(true_betas) + error


@pytest.mark.parametrize("alpha", [0, 0.5, 1])
def test_linear_glmnet(benchmark, alpha):
    m = ElasticNet(alpha=alpha)
    benchmark(m.fit, X, y)


@pytest.mark.parametrize("alpha", [0, 0.5, 1])
def test_linear_elnetpy(benchmark, alpha):
    m2 = Elnet(alpha=alpha)
    benchmark(m2.fit, X, y)
