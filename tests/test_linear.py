"Test linear elastic net"

import numpy as np

from elnetpy.linear import Elnet


def test_elnet():
    error = np.random.normal(loc=0, scale=1, size=100)
    X = np.random.normal(loc=5, scale=2, size=(100, 4))
    true_betas = np.array([1, -2, 0.5, 1])
    y = X.dot(true_betas) + error
    lambdas = np.array([0, 0.5, 1, 2])
    elnet = Elnet(lambdas=lambdas)
    elnet.fit(X, y)
