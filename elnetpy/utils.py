"Helper functions for elastic net"

import numpy as np
from scipy.stats import zscore


def standardize_inputs(X, y, return_means_stds=False):
    if return_means_stds:
        X_means = X.mean(axis=0)
        X_stds = X.std(axis=0)
        y_mean = np.mean(y)
        y_std = np.std(y)
    X = zscore(X)
    y = zscore(y)
    return (X, y, X_means, X_stds, y_mean, y_std) if return_means_stds else (X, y)


def destandardize_coefs(coefs_mat, X_means, X_stds, y_mean, y_std):
    """Destandardize elastic net (standardized) coefficients

    :param coefs_mat: Matrix of coefficients of shape (n_features, n_lambdas)
    :type coefs_mat: numpy.array
    :param X_stds: Means of the features of shape (n_features,)
    :type X_std: numpy.array
    :param X_stds: Standard deviations of the features of shape (n_features,)
    :type X_std: numpy.array
    :param y_mean: Mean of the response
    :type y_mean: float
    :param y_std: Standard deviation of the response
    :type y_std: float

    :return: Coefficient matrix of shape (n_features, n_lambdas) and intercepts vector
    of shape (n_lambdas,)
    :rtype: tuple
    """
    coefs_mat_destd = (coefs_mat * y_std) / X_stds[:, None]
    intercepts = y_mean - coefs_mat_destd.T.dot(X_means)
    return coefs_mat_destd, intercepts


def get_lambda_path(X, y, alpha, min_lambda_ratio, n_lambda):
    n_obs = X.shape[0]
    lambda_max = np.max(np.abs(X.T.dot(y))) / n_obs
    lambda_min = lambda_max * min_lambda_ratio
    # sequence on the log scale
    lambda_path = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), n_lambda))
    return lambda_path
