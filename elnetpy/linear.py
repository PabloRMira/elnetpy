"Linear Elastic Net"

import numpy as np
from sklearn.base import BaseEstimator

from elnetpy import linear_elnet
from elnetpy.utils import standardize_inputs, destandardize_coefs, get_lambda_path


class Elnet(BaseEstimator):
    """Elastic Net estimator for linear models"""

    def __init__(
        self,
        alpha=1,
        lambdas=None,
        n_lambda=100,
        min_lambda_ratio=1e-4,
        n_jobs=1,
        tol=1e-7,
        max_iter=1e5,
    ):
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambda = n_lambda
        self.min_lambda_ratio = min_lambda_ratio
        self.n_jobs = n_jobs
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):

        self._fit(X, y)

        return self

    def _fit(self, X, y):

        # for matrices to be passed by reference with pybind11
        # we need Fortran column major storage order
        _X = X.astype(dtype="float64", order="F", copy=True)
        _y = y.astype(dtype="float64", copy=True)

        _X, _y, X_means, X_stds, y_mean, y_std = standardize_inputs(
            _X, _y, return_means_stds=True
        )

        if self.lambdas is None:
            self.lambda_path_ = get_lambda_path(
                _X, _y, y_std, self.alpha, self.min_lambda_ratio, self.n_lambda
            )
        else:
            self.lambda_path_ = (
                self.lambdas
                if isinstance(self.lambdas, np.ndarray)
                else np.array([self.lambdas])
            ).astype(dtype="float64") / y_std

        # get standardized coefficients
        coefs_mat = linear_elnet(_X, _y, self.lambda_path_, self.tol, self.max_iter)

        # destandardize coefficients and get intercepts
        self.coef_path_, self.intercept_path_ = destandardize_coefs(
            coefs_mat, X_means, X_stds, y_mean, y_std
        )

        return self
