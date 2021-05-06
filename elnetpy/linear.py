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
        self._validate_input(
            alpha, lambdas, n_lambda, min_lambda_ratio, n_jobs, tol, max_iter
        )
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
            lambda_path_std = get_lambda_path(
                _X, _y, self.alpha, self.min_lambda_ratio, self.n_lambda
            )
            self.lambda_path_ = lambda_path_std * y_std
        else:
            lambda_path_std = (
                self.lambdas
                if isinstance(self.lambdas, np.ndarray)
                else np.array([self.lambdas])
            ).astype(dtype="float64")
            self.lambda_path_ = lambda_path_std
            lambda_path_std /= y_std

        # get standardized coefficients
        early_stopping = True
        coefs_mat = linear_elnet(
            _X, _y, lambda_path_std, self.alpha, early_stopping, self.tol, self.max_iter
        )

        # destandardize coefficients and get intercepts
        self.coef_path_, self.intercept_path_ = destandardize_coefs(
            coefs_mat, X_means, X_stds, y_mean, y_std
        )

        # truncate lambda sequence for stopping criterion
        self.lambda_path_ = self.lambda_path_[0 : self.coef_path_.shape[1]]

        return self

    @staticmethod
    def _validate_input(
        alpha, lambdas, n_lambda, min_lambda_ratio, n_jobs, tol, max_iter
    ):
        if alpha < 0 or alpha > 1:
            raise ValueError(
                f"""Alpha should be between 0 and 1 inclusively.
                Your input alpha is {alpha}"""
            )
        if isinstance(lambdas, str):
            raise ValueError("lambdas should be either None, numeric or numpy array")
        if n_lambda <= 5:
            raise ValueError("n_lambda should be greater than 5")
        if min_lambda_ratio <= 0 or min_lambda_ratio >= 1:
            raise ValueError(
                f"""min_lambda_ratio should be between 0 and 1 exclusively.
                You input min_lambda_ratio is {min_lambda_ratio}"""
            )
        if not isinstance(n_jobs, int):
            raise ValueError(
                f"""n_jobs should be integer valued.
                Your input n_jobs is {n_jobs}"""
            )
        if n_jobs < -1 or n_jobs == 0:
            raise ValueError(
                f"""n_jobs should be greater than -1 and not 0.
                Your input n_jobs is {n_jobs}"""
            )
        if tol <= 0 or tol >= 1:
            raise ValueError(
                f"""tol should be between 0 and 1 exclusively.
                Your input tol is {tol}"""
            )
        if max_iter < 100:
            raise ValueError(
                f"""max_iter should be greater than or equal 100
                Your input max_iter is {max_iter}"""
            )
