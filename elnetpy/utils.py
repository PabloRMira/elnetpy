"Helper functions for elastic net"

import numpy as np
from scipy.stats import zscore


def standardize_inputs(X, y, lambdas):
    X = zscore(X)
    y_std = np.std(y)
    y = zscore(y)
    lambdas = lambdas / y_std
    return X, y, lambdas
